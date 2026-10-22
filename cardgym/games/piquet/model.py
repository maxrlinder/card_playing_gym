"""Piquet actor on the shared trunk. There is no critic: the actor's value head is the baseline.

Input. A token is its fields (one packed lookup) plus a summary of its 32
card codes from :class:`CardGrid`:

* each card starts as the embedding of its code plus a rank embedding (ranks
  are not interchangeable: values, the 10-and-up sets, Ace high);
* grid blocks on the 4 x 8 suit-by-rank grid mix a convolution along each
  suit's ranks (sequences; zero-padded, since Piquet's ranks do not wrap)
  with each rank's mean over suits (sets). Both are equivariant to suit
  permutations, the game's own symmetry without trumps, so combination
  structure is learned once for every suit;
* a per-card linear map flattens the grid into the token embedding, keeping
  card identity.

Output. ``exchange`` and ``play`` point at cards: each card's score is an
MLP of the readout state plus that card's grid features and an identity row
(additive attention), so the head sees card-level detail; ``exchange`` adds
one logit for "stop". ``declare`` is a linear map to its two actions. The
heads reuse the grid features the embedding computed (``pop_context``).

Value and predictions, all from the readout states of the actor's own
stream: the per-seat value (the PPO baseline), and auxiliary heads for where
each unknown card is (opponent's hand, its discards, the talon), the seat's
tricks this deal, who wins each declaration category, both seats' points
this deal, repique / pique / capot, who wins the current trick, the
opponent's suit lengths after the exchange, the next card the opponent
plays, and every card's fate (the trick it falls in and whether it wins it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cardgym.games.base import PolicyModel
from cardgym.nn.model import value_mlp
from cardgym.nn.trunk import TrunkConfig, ValueEmbeddingMode, init_weights

from .cards import CARD_RANKS, NUM_CARDS, NUM_RANKS, NUM_SUITS
from .engine import HEADS, TRICKS
from .tokens import (
    BELIEF_CLASSES,
    CARD_START,
    CARD_VOCAB,
    FIELD_VOCAB,
    LEGACY_FIELD_VOCAB,
    NUM_FIELDS,
    PARTIE_FIELDS,
)

PLAYERS = 2
CATEGORIES = 3
OUTCOME_CLASSES = 3  # me, the opponent, nobody (declarations); none, me, the opponent (bonuses)
BONUSES = 3  # repique, pique, capot
SUIT_LENGTHS = NUM_RANKS + 1  # cards of a suit in a hand: 0-8
# A card's fate this deal: played to trick t and won it (2t + 1) or lost it
# (2t), or never played (discarded, or left in the talon).
FATE_CLASSES = 2 * TRICKS + 1
NEVER_PLAYED = 2 * TRICKS
# Auxiliary outputs, packed into one projection: name -> width.
AUXILIARY = {
    "belief": NUM_CARDS * BELIEF_CLASSES,
    "tricks": TRICKS + 1,
    "declarations": CATEGORIES * OUTCOME_CLASSES,
    "points": PLAYERS,
    "bonus": BONUSES * OUTCOME_CLASSES,
    "trick": PLAYERS,
    "opp_suits": NUM_SUITS * SUIT_LENGTHS,
    "opp_next": NUM_CARDS,
    "fate": NUM_CARDS * FATE_CLASSES,
}


@dataclass(frozen=True)
class PiquetModelConfig:
    """The [model] section. The defaults are the 16-layer, 9.1M-parameter actor."""

    d_model: int = 256
    n_layers: int = 16
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 512
    # 64 (or 0) keeps the fused Metal trunk available.
    per_layer_dim: int = 64
    value_embeddings: ValueEmbeddingMode = "alternating"
    # Card grid: per-card width, block MLP width, blocks, rank-convolution taps.
    card_width: int = 32
    card_hidden: int = 64
    card_layers: int = 2
    card_kernel: int = 3
    # Card scorer width.
    score_dim: int = 64

    def trunk(self) -> TrunkConfig:
        return TrunkConfig(
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            per_layer_dim=self.per_layer_dim,
            value_embeddings=self.value_embeddings,
            input_norm=True,
        )

    def validate(self) -> None:
        self.trunk().validate()
        if min(self.card_width, self.card_hidden, self.score_dim) < 1 or self.card_layers < 0:
            raise ValueError("[model] card and score widths must be positive.")
        if self.card_kernel < 1 or self.card_kernel % 2 == 0 or self.card_kernel > 2 * NUM_RANKS - 1:
            raise ValueError("[model] card_kernel must be odd and at most 15.")


class GridNorm(nn.Module):
    """RMSNorm with a gain, through the fused ``F.rms_norm`` (fp32 statistics)."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class GridBlock(nn.Module):
    """Residual mix of rank neighbours (zero-padded) and same-rank suits on the 4 x 8 card grid.

    The convolution runs as one block-Toeplitz GEMM per suit row and the suit
    mean and its broadcast as [1, 4] and [4, 1] matmuls: on MPS, reductions
    and broadcasts over a middle axis are far slower than GEMMs (see Rummy's
    GridBlock, which this follows without the wrap-around). Under bf16
    autocast on MPS the whole grid runs as fused kernels instead
    (:mod:`cardgym.nn.metal_grid`, zero-padded variant).
    """

    def __init__(self, width: int, hidden: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.norm = GridNorm(width)
        self.along_rank = nn.Conv1d(width, hidden, kernel)
        self.across_suits = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width)
        half = kernel // 2
        # shifts[k, i, o] = 1 when input rank i is tap k of output rank o.
        shifts = torch.zeros(kernel, NUM_RANKS, NUM_RANKS)
        for tap in range(kernel):
            for rank in range(NUM_RANKS):
                source = rank + tap - half
                if 0 <= source < NUM_RANKS:
                    shifts[tap, source, rank] = 1.0
        self.register_buffer("shifts", shifts, persistent=False)
        self.register_buffer("pool", torch.full((1, NUM_SUITS), 1.0 / NUM_SUITS), persistent=False)
        self.register_buffer("spread", torch.ones(NUM_SUITS, 1), persistent=False)

    def toeplitz(self) -> torch.Tensor:
        """The convolution as a [8 * width, 8 * hidden] matrix acting on one suit row."""

        weight = self.along_rank.weight  # [hidden, width, kernel]
        hidden, width, _ = weight.shape
        return torch.einsum("kio,hwk->iwoh", self.shifts.to(weight.dtype), weight).reshape(
            NUM_RANKS * width, NUM_RANKS * hidden
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:  # [N, suits, ranks, width]
        count, _, _, width = grid.shape
        hidden = self.out.in_features
        rows = self.norm(grid).reshape(count, NUM_SUITS, NUM_RANKS * width)
        along = F.linear(rows, self.toeplitz().t(), self.along_rank.bias.repeat(NUM_RANKS))
        pooled = (self.pool.to(rows.dtype) @ rows).view(count, NUM_RANKS, width)
        across = self.across_suits(pooled).view(count, 1, NUM_RANKS * hidden)
        mixed = along + self.spread.to(across.dtype) @ across
        return grid + self.out(F.gelu(mixed).view(count, NUM_SUITS, NUM_RANKS, hidden))


class CardGrid(nn.Module):
    """Card codes [..., 32] -> per-card features [..., 32, width] and a token summary [..., d]."""

    def __init__(self, config: PiquetModelConfig) -> None:
        super().__init__()
        width = config.card_width
        self.width = width
        self.code = nn.Embedding(CARD_VOCAB, width)
        self.rank = nn.Embedding(NUM_RANKS, width)
        self.blocks = nn.ModuleList(
            GridBlock(width, config.card_hidden, config.card_kernel) for _ in range(config.card_layers)
        )
        self.norm = GridNorm(width)
        self.summary = nn.Linear(NUM_CARDS * width, config.d_model, bias=False)
        self.hidden = config.card_hidden
        self.eps = self.norm.eps
        # The fused kernels (cardgym.nn.metal_grid) under MPS bf16 autocast:
        # 8 ranks, zero-padded. Set by enable_metal_trunk, like the trunk's.
        self.ranks, self.circular = NUM_RANKS, False
        self.fused = False
        self._fused_key: tuple | None = None
        self._fused_operands: dict | None = None
        for module in (self.code, self.rank, *self.blocks):
            module.apply(init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.out.weight.mul_(1.0 / max(len(self.blocks), 1) ** 0.5)
        # Comparable scale to the sum of the field rows.
        nn.init.normal_(self.summary.weight, std=0.02 * (NUM_FIELDS / (NUM_CARDS * width)) ** 0.5)
        self.register_buffer("card_ranks", torch.as_tensor(CARD_RANKS), persistent=False)

    def uses_fused(self, device: torch.device) -> bool:
        return self.fused and len(self.blocks) > 0 and _fused(device)

    def cards(self, codes: torch.Tensor) -> torch.Tensor:
        lead = codes.shape[:-1]
        if self.uses_fused(codes.device):
            from cardgym.nn.metal_grid import FusedCardGrid

            params = [self.code.weight, self.rank.weight]
            for block in self.blocks:
                params += [
                    block.norm.weight, block.along_rank.weight, block.along_rank.bias,
                    block.across_suits.weight, block.out.weight, block.out.bias,
                ]
            params.append(self.norm.weight)
            flat = codes.reshape(-1, NUM_CARDS).long().contiguous()
            save = torch.is_grad_enabled() and any(p.requires_grad for p in params)
            return FusedCardGrid.apply(self, flat, save, *params).reshape(*lead, NUM_CARDS, self.width)
        grid = self.code(codes.long()) + self.rank(self.card_ranks)
        grid = grid.reshape(-1, NUM_SUITS, NUM_RANKS, self.width)
        for block in self.blocks:
            grid = block(grid)
        return self.norm(grid).reshape(*lead, NUM_CARDS, self.width)

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cards = self.cards(codes)
        return cards, self.summary(cards.flatten(-2))


def _fused(device: torch.device) -> bool:
    """Fused Metal kernels apply: MPS under bf16 autocast (training and rollout alike)."""

    return (
        device.type == "mps"
        and torch.is_autocast_enabled("mps")
        and torch.get_autocast_dtype("mps") == torch.bfloat16
    )


def _bag(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = ids.reshape(-1, ids.shape[-1])
    return F.embedding_bag(flat, weight, mode="sum").view(*ids.shape[:-1], weight.shape[-1])


def migrate_field_table(old: torch.Tensor) -> torch.Tensor:
    """A field table of the pre-partie layout (LEGACY_FIELD_VOCAB) in the current one.

    Every row of a partie field becomes that field's row 0 (the only one a
    single-deal model has seen) and the new lead field adds zero: the model
    computes exactly what it did, blind to the score, and a fine-tune on
    parties starts from it and learns what the score fields mean.
    """

    new = old.new_zeros(sum(FIELD_VOCAB), old.shape[1])
    old_at = new_at = 0
    for field, (before, after) in enumerate(zip(LEGACY_FIELD_VOCAB, FIELD_VOCAB)):
        rows = old[old_at : old_at + before]
        if field in PARTIE_FIELDS:
            new[new_at : new_at + after] = rows[0]
        else:
            assert before == after, "Only the partie fields changed size."
            new[new_at : new_at + after] = rows
        old_at += before
        new_at += after
    return new


class PiquetEmbedding(nn.Module):
    """Token fields plus the card-grid summary; keeps the latest per-card features for the heads."""

    def __init__(self, config: PiquetModelConfig, ple_width: int, value_width: int) -> None:
        super().__init__()
        self.kept: torch.Tensor | None = None
        offsets = torch.tensor([sum(FIELD_VOCAB[:i]) for i in range(NUM_FIELDS)], dtype=torch.long)
        self.register_buffer("offsets", offsets, persistent=False)
        rows = sum(FIELD_VOCAB)
        self.fields = nn.Embedding(rows, config.d_model)
        init_weights(self.fields)
        # Conditioning starts at zero: enabling it leaves the function unchanged.
        self.ple_fields = nn.Embedding(rows, ple_width) if ple_width else None
        self.value_fields = nn.Embedding(rows, value_width) if value_width else None
        for table in (self.ple_fields, self.value_fields):
            if table is not None:
                nn.init.zeros_(table.weight)
        self.grid = CardGrid(config)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before the partie fields load blind to them.
        for name in ("fields", "ple_fields", "value_fields"):
            key = f"{prefix}{name}.weight"
            if key in state_dict and state_dict[key].shape[0] == sum(LEGACY_FIELD_VOCAB):
                state_dict[key] = migrate_field_table(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, tokens: torch.Tensor):
        tokens = tokens.long()
        ids = tokens[..., :NUM_FIELDS] + self.offsets
        cards, summary = self.grid(tokens[..., CARD_START:])
        self.kept = cards
        tables = [self.fields.weight] + [
            table.weight for table in (self.ple_fields, self.value_fields) if table is not None
        ]
        if _fused(tokens.device):
            # One fused lookup per table, one GEMM for all their gradients (cardgym.nn.metal_bag).
            from cardgym.nn.metal_bag import FieldBags

            sums = [value.view(*ids.shape[:-1], -1) for value in FieldBags.apply(ids.reshape(-1, NUM_FIELDS), *tables)]
        else:
            sums = [_bag(ids, table) for table in tables]
        sums = iter(sums)
        return (
            next(sums) + summary,
            None if self.ple_fields is None else next(sums),
            None if self.value_fields is None else next(sums),
        )


class CardScorer(nn.Module):
    """One logit per card: additive attention of the readout state against the card's features."""

    def __init__(self, d_model: int, card_width: int, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, dim)
        self.card = nn.Linear(card_width, dim, bias=False)
        self.card_id = nn.Embedding(NUM_CARDS, dim)
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)
        self.apply(init_weights)

    def forward(self, hidden: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:  # [N, d], [N, 32, w]
        mixed = self.query(hidden)[:, None, :] + self.card(grid) + self.card_id.weight.to(grid.dtype)
        return self.out(F.gelu(self.hidden(self.norm(mixed)))).squeeze(-1)


class PiquetActor(PolicyModel):
    policy_heads = HEADS

    def __init__(self, config: PiquetModelConfig) -> None:
        super().__init__(config.trunk())
        self.config = config
        trunk = self.trunk_config
        d = config.d_model
        self.embedding = PiquetEmbedding(config, trunk.ple_width, trunk.value_width)
        self.exchange_scorer = CardScorer(d, config.card_width, config.score_dim)
        self.play_scorer = CardScorer(d, config.card_width, config.score_dim)
        self.stop_head = nn.Linear(d, 1)
        self.declare_head = nn.Linear(d, 2)
        self.value_head = value_mlp(d, PLAYERS)
        self.auxiliary_head = nn.Linear(d, sum(AUXILIARY.values()))
        for module in (self.stop_head, self.declare_head, self.auxiliary_head):
            init_weights(module)

    def embed(self, tokens: torch.Tensor):
        return self.embedding(tokens)

    def enable_metal_trunk(self, max_length: int | None = None) -> None:
        super().enable_metal_trunk(max_length)
        self.embedding.grid.fused = metal_grid_supported(self.config)

    def pop_context(self) -> torch.Tensor | None:
        cards, self.embedding.kept = self.embedding.kept, None
        return cards

    def policy_logits(self, hidden, head, features=None, tokens=None, context=None):
        if head == "declare":
            return self.declare_head(hidden)
        if head not in ("exchange", "play"):
            raise ValueError(f"Unknown piquet policy head {head!r}.")
        if context is None:
            if tokens is None:
                raise ValueError("Card heads need the readout tokens or their grid features.")
            context = self.embedding.grid.cards(tokens[..., CARD_START:])
        if head == "play":
            return self.play_scorer(hidden, context)
        return torch.cat([self.exchange_scorer(hidden, context), self.stop_head(hidden)], dim=-1)

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)

    def auxiliary(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every auxiliary prediction from one projection: name -> [..., width]."""

        return dict(zip(AUXILIARY, self.auxiliary_head(hidden).split(list(AUXILIARY.values()), dim=-1)))


def metal_grid_supported(config: PiquetModelConfig) -> bool:
    from cardgym.nn.metal_grid import supported

    return config.card_layers > 0 and supported(
        config.card_width, config.card_hidden, config.card_kernel, NUM_RANKS, circular=False
    )
