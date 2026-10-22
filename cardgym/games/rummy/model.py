"""Rummy actor and oracle critic on the shared trunk.

Input. A token is its fields (one packed lookup, as in plump) plus a summary
of its 52 per-card codes computed by :class:`CardGrid`. On a decision token
the codes are the viewer's state (where both copies are, who tabled them,
layable); on an opponent's turn token, what that turn did with each card type
(taken, melded, laid off, discarded). One grid reads both:

* each card type starts as the embedding of its code plus a rank embedding
  (points break the rank symmetry the rules otherwise have);
* grid blocks on the 4 x 13 suit-by-rank grid mix a circular convolution
  along the rank cycle (runs are cyclic) with each rank's mean over suits
  (groups). Both are equivariant to suit permutations and rank rotations, the
  rules' own symmetries, so meld structure is learned once for every suit and
  every rank;
* a per-card linear map flattens the grid into the token embedding, which
  keeps card identity (and therefore injectivity) rather than pooling it away.

The grid runs once per token, before the trunk. It is the token's input, so a
cached key never goes stale: each position keeps the cards of its own moment.

Output. Both policy heads score the engine's candidate list. A candidate's
embedding sums its kind, its moved cards, (for a lay-off) the target meld's
cards, and (for a pickup) every card it takes from the pile, each represented by its grid features at the decision
point plus an identity row. The readout hidden state is added and an MLP
scores the result (additive attention), so the head sees card-level detail
the pooled token summary may blur. The heads read decision tokens and reuse
the grid features the embedding just computed for them (``pop_context``)
rather than encoding the state twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cardgym.games.base import CriticModel, PolicyModel
from cardgym.nn.model import value_mlp
from cardgym.nn.trunk import TrunkConfig, ValueEmbeddingMode, init_weights

from .cards import CARD_RANKS, NO_CARD, NUM_CARDS, NUM_RANKS, NUM_SUITS
from .engine import MAX_PLAYERS, NUM_ACTION_KINDS
from .selfplay import MAX_MELD_CARDS
from .tokens import CARD_START, CARD_VOCAB, FIELD_VOCAB, NUM_FIELDS

OPPONENTS = MAX_PLAYERS - 1
COUNT_CLASSES = 3  # copies of a card type in one hand: 0, 1, 2
NO_WINNER = MAX_PLAYERS  # winner class for a round without one


@dataclass(frozen=True)
class RummyModelConfig:
    """The [model] section. Shapes are a starting point, not a tuned recipe."""

    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 768
    # 64 (or 0) keeps the fused Metal trunk available.
    per_layer_dim: int = 64
    value_embeddings: ValueEmbeddingMode = "alternating"
    # Card grid: per-card width, block MLP width, blocks, rank-convolution taps.
    ledger_width: int = 32
    ledger_hidden: int = 64
    ledger_layers: int = 2
    ledger_kernel: int = 5
    # Candidate scorer width.
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
        if self.ledger_width < 1 or self.ledger_hidden < 1 or self.ledger_layers < 0 or self.score_dim < 1:
            raise ValueError("[model] ledger and score widths must be positive.")
        if self.ledger_kernel < 1 or self.ledger_kernel % 2 == 0 or self.ledger_kernel > NUM_RANKS:
            raise ValueError("[model] ledger_kernel must be odd and at most 13.")


class FusedRMSNorm(nn.Module):
    """RMSNorm through ``F.rms_norm`` (same function and parameters as the trunk's).

    The trunk's op-by-op RMSNorm backward, rerun from MPS's graph cache at a
    few hundred rows (the policy head's decisions), returned all-NaN input
    gradients in 1 of 4 policy passes; the fused op returned none in 48 and
    matches a CPU fp32 reference to bf16 accuracy.
    """

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class GridBlock(nn.Module):
    """Residual mix of rank neighbours (circular) and same-rank suits on the card grid.

    The layer is a circular convolution along each suit's 13 ranks plus a
    linear map of each rank's mean over suits, then GELU and a projection.
    Its execution is shaped for MPS, where reducing or broadcasting over a
    middle axis is pathologically slow (a [N, 4, 13, 32] suit mean measured
    33 ms against 1.3 ms as a matmul): the convolution runs as one
    block-circulant GEMM per suit row, built from the Conv1d weights each
    call, and the suit pooling and its broadcast back are [1, 4] and [4, 1]
    matmuls, which keeps their backward passes GEMMs too. Same function and
    parameters as the direct form; 3.4x faster forward and backward. Under
    bf16 autocast on MPS the whole grid runs as fused kernels instead
    (:mod:`cardgym.nn.metal_grid`).
    """

    def __init__(self, width: int, hidden: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.norm = FusedRMSNorm(width)
        self.along_rank = nn.Conv1d(width, hidden, kernel)
        self.across_suits = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width)
        # shifts[k, i, o] = 1 when input rank i is tap k of output rank o (circular padding).
        half = kernel // 2
        shifts = torch.zeros(kernel, NUM_RANKS, NUM_RANKS)
        for tap in range(kernel):
            for rank in range(NUM_RANKS):
                shifts[tap, (rank + tap - half) % NUM_RANKS, rank] = 1.0
        self.register_buffer("shifts", shifts, persistent=False)
        self.register_buffer("pool", torch.full((1, NUM_SUITS), 1.0 / NUM_SUITS), persistent=False)
        self.register_buffer("spread", torch.ones(NUM_SUITS, 1), persistent=False)

    def circulant(self, weight: torch.Tensor | None = None) -> torch.Tensor:
        """The convolution as a [13 * width, 13 * hidden] matrix acting on one suit row."""

        weight = self.along_rank.weight if weight is None else weight  # [hidden, width, kernel]
        hidden, width, _ = weight.shape
        return torch.einsum("kio,hwk->iwoh", self.shifts.to(weight.dtype), weight).reshape(
            NUM_RANKS * width, NUM_RANKS * hidden
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:  # [N, suits, ranks, width]
        count, _, _, width = grid.shape
        hidden = self.out.in_features
        y = self.norm(grid)
        rows = y.reshape(count, NUM_SUITS, NUM_RANKS * width)
        along = F.linear(rows, self.circulant().t(), self.along_rank.bias.repeat(NUM_RANKS))
        pooled = (self.pool.to(rows.dtype) @ rows).view(count, NUM_RANKS, width)
        across = self.across_suits(pooled).view(count, 1, NUM_RANKS * hidden)
        mixed = along + self.spread.to(across.dtype) @ across
        return grid + self.out(F.gelu(mixed).view(count, NUM_SUITS, NUM_RANKS, hidden))


class CardGrid(nn.Module):
    """Card codes [..., 52] -> per-card features [..., 52, width] and a token summary [..., d]."""

    def __init__(self, config: RummyModelConfig) -> None:
        super().__init__()
        width = config.ledger_width
        self.width = width
        self.code = nn.Embedding(CARD_VOCAB, width)
        self.rank = nn.Embedding(NUM_RANKS, width)
        self.blocks = nn.ModuleList(
            GridBlock(width, config.ledger_hidden, config.ledger_kernel) for _ in range(config.ledger_layers)
        )
        self.norm = FusedRMSNorm(width)
        self.summary = nn.Linear(NUM_CARDS * width, config.d_model, bias=False)
        self.hidden = config.ledger_hidden
        self.eps = self.norm.eps
        # The fused kernels' grid shape: 13 ranks on a cycle.
        self.ranks, self.circular = NUM_RANKS, True
        # Fused Metal kernels (metal_grid) under MPS bf16 autocast; set by
        # enable_metal_trunk, like the trunk's.
        self.fused = False
        self._fused_key: tuple | None = None
        self._fused_operands: dict | None = None
        for module in (self.code, self.rank, *self.blocks):
            module.apply(init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.out.weight.mul_(1.0 / max(len(self.blocks), 1) ** 0.5)
        # Comparable scale to the sum of the event-field rows.
        nn.init.normal_(self.summary.weight, std=0.02 * (NUM_FIELDS / (NUM_CARDS * width)) ** 0.5)
        self.register_buffer("card_ranks", torch.as_tensor(CARD_RANKS, dtype=torch.long), persistent=False)

    def uses_fused(self, device: torch.device) -> bool:
        return (
            self.fused
            and device.type == "mps"
            and len(self.blocks) > 0
            and torch.is_autocast_enabled("mps")
            and torch.get_autocast_dtype("mps") == torch.bfloat16
        )

    def cards(self, ledger: torch.Tensor) -> torch.Tensor:
        lead = ledger.shape[:-1]
        if self.uses_fused(ledger.device):
            from cardgym.nn.metal_grid import FusedCardGrid

            params = [self.code.weight, self.rank.weight]
            for block in self.blocks:
                params += [
                    block.norm.weight, block.along_rank.weight, block.along_rank.bias,
                    block.across_suits.weight, block.out.weight, block.out.bias,
                ]
            params.append(self.norm.weight)
            flat = ledger.reshape(-1, NUM_CARDS).long().contiguous()
            save = torch.is_grad_enabled() and any(p.requires_grad for p in params)
            return FusedCardGrid.apply(self, flat, save, *params).reshape(*lead, NUM_CARDS, self.width)
        grid = self.code(ledger.long()) + self.rank(self.card_ranks)
        grid = grid.reshape(-1, NUM_SUITS, NUM_RANKS, self.width)
        for block in self.blocks:
            grid = block(grid)
        return self.norm(grid).reshape(*lead, NUM_CARDS, self.width)

    def forward(self, ledger: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cards = self.cards(ledger)
        return cards, self.summary(cards.flatten(-2))


def _bag(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    flat = ids.reshape(-1, ids.shape[-1])
    return F.embedding_bag(flat, weight, mode="sum").view(*ids.shape[:-1], weight.shape[-1])


class RummyEmbedding(nn.Module):
    """Token fields plus the card-grid summary; optionally keeps the per-card features.

    With ``keep_cards`` the latest forward's per-card features [..., 52, w]
    stay available (once) for the policy heads.
    """

    def __init__(
        self, config: RummyModelConfig, ple_width: int, value_width: int, *, keep_cards: bool = False
    ) -> None:
        super().__init__()
        self.keep_cards = keep_cards
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

    def forward(self, tokens: torch.Tensor):
        tokens = tokens.long()
        ids = tokens[..., :NUM_FIELDS] + self.offsets
        cards, summary = self.grid(tokens[..., CARD_START:])
        if self.keep_cards:
            self.kept = cards
        tables = [self.fields.weight] + [
            table.weight for table in (self.ple_fields, self.value_fields) if table is not None
        ]
        if self.grid.uses_fused(tokens.device):
            # One fused lookup for every table (cardgym.nn.metal_bag).
            from cardgym.nn.metal_bag import FieldBags

            sums = [
                value.view(*ids.shape[:-1], -1)
                for value in FieldBags.apply(ids.reshape(-1, NUM_FIELDS), *tables)
            ]
        else:
            sums = [_bag(ids, table) for table in tables]
        sums = iter(sums)
        fields = next(sums) + summary
        return (
            fields,
            None if self.ple_fields is None else next(sums),
            None if self.value_fields is None else next(sums),
        )


class CandidateScorer(nn.Module):
    """Additive-attention score of each candidate action against the readout state."""

    def __init__(self, d_model: int, card_width: int, dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, dim)
        self.card = nn.Linear(card_width, dim, bias=False)
        self.card_id = nn.Embedding(NUM_CARDS, dim)
        self.target = nn.Linear(card_width, dim, bias=False)
        self.target_id = nn.Embedding(NUM_CARDS, dim)
        self.kind = nn.Embedding(NUM_ACTION_KINDS, dim)
        self.size = nn.Embedding(MAX_MELD_CARDS + 1, dim)
        # Draw candidates: every card a pickup takes from the discard pile.
        self.taken = nn.Linear(card_width, dim, bias=False)
        self.taken_id = nn.Embedding(NUM_CARDS, dim)
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)
        self.apply(init_weights)
        # Inference on MPS runs as one Metal kernel (metal_scorer); set by
        # enable_metal_trunk. Training keeps this module's autograd path.
        self.fused = False
        self._fused_key: tuple | None = None
        self._fused_weights: dict | None = None

    def forward(
        self,
        hidden: torch.Tensor,
        grid: torch.Tensor,
        kind: torch.Tensor,
        cards: torch.Tensor,
        target: torch.Tensor,
        taken: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.fused and hidden.device.type == "mps" and not torch.is_grad_enabled():
            from .metal_scorer import score

            return score(self, hidden, grid, kind, cards, target, taken)
        rows, width = kind.shape
        dtype = grid.dtype
        slots = 2 if taken is None else 3
        # Copies of each card type per slot (moved, lay-off target, taken):
        # one scatter into [N, K, slots, 52]; NO_CARD entries add zero.
        ids = torch.cat([cards, target], dim=-1).long()
        present = ids != NO_CARD
        offsets = torch.arange(2, device=ids.device).repeat_interleave(cards.shape[-1]) * NUM_CARDS
        counts = torch.zeros(rows, width, slots * NUM_CARDS, device=ids.device, dtype=dtype)
        counts.scatter_add_(2, torch.where(present, ids + offsets, 0), present.to(dtype))
        if taken is not None:
            counts[..., 2 * NUM_CARDS :] = taken.to(dtype)
        # Sum each slot's card features first, then project: by linearity the
        # same as projecting all 52 cards per row, at a fraction of the
        # traffic. All slots go through one bmm and one GEMM each for the
        # features and the identity rows.
        summed = torch.bmm(counts.view(rows, width * slots, NUM_CARDS), grid).view(rows, width, -1)
        projections = (self.card.weight, self.target.weight, self.taken.weight)[:slots]
        identities = (self.card_id.weight, self.target_id.weight, self.taken_id.weight)[:slots]
        candidate = (
            F.linear(summed, torch.cat(projections, dim=1))
            + counts @ torch.cat(identities).to(dtype)
            + self.kind(kind.long())
            + self.size(present[..., : cards.shape[-1]].sum(-1))
        )
        mixed = self.norm(self.query(hidden)[:, None, :] + candidate)
        return self.out(F.gelu(self.hidden(mixed))).squeeze(-1)


class RummyActor(PolicyModel):
    policy_heads = ("draw", "play")

    def __init__(self, config: RummyModelConfig) -> None:
        super().__init__(config.trunk())
        self.config = config
        trunk = self.trunk_config
        d = config.d_model
        self.embedding = RummyEmbedding(config, trunk.ple_width, trunk.value_width, keep_cards=True)
        self.scorer = CandidateScorer(d, config.ledger_width, config.score_dim)
        self.value_head = value_mlp(d, MAX_PLAYERS)
        # Beliefs: copies of each card type in each opponent's hand.
        self.hands_head = nn.Linear(d, OPPONENTS * NUM_CARDS * COUNT_CLASSES)
        self.winner_head = nn.Linear(d, MAX_PLAYERS + 1)
        self.penalty_head = nn.Linear(d, MAX_PLAYERS)
        for module in (self.hands_head, self.winner_head, self.penalty_head):
            init_weights(module)

    def embed(self, tokens: torch.Tensor):
        return self.embedding(tokens)

    def enable_metal_trunk(self, max_length: int | None = None) -> None:
        super().enable_metal_trunk(max_length)
        self.embedding.grid.fused = metal_grid_supported(self.config)
        from .metal_scorer import supported

        self.scorer.fused = supported(self.scorer)

    def pop_context(self) -> torch.Tensor | None:
        cards, self.embedding.kept = self.embedding.kept, None
        return cards

    def policy_logits(self, hidden, head, features=None, tokens=None, context=None):
        if features is None or (tokens is None and context is None):
            raise ValueError("Rummy policy heads score candidate lists at a readout token.")
        if head not in self.policy_heads:
            raise ValueError(f"Unknown rummy policy head {head!r}.")
        grid = self.embedding.grid.cards(tokens[..., CARD_START:]) if context is None else context
        return self.scorer(
            hidden, grid, features["kind"], features["cards"], features["target"], features.get("taken")
        )

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)


def metal_grid_supported(config: RummyModelConfig) -> bool:
    from cardgym.nn.metal_grid import supported

    return config.ledger_layers > 0 and supported(config.ledger_width, config.ledger_hidden, config.ledger_kernel)


class RummyCritic(CriticModel):
    def __init__(self, config: RummyModelConfig) -> None:
        super().__init__(config.trunk())
        self.config = config
        trunk = self.trunk_config
        d = config.d_model
        self.embedding = RummyEmbedding(config, trunk.ple_width, trunk.value_width)
        self.value_head = value_mlp(d, MAX_PLAYERS)
        self.winner_head = nn.Linear(d, MAX_PLAYERS + 1)
        self.penalty_head = nn.Linear(d, MAX_PLAYERS)
        for module in (self.winner_head, self.penalty_head):
            init_weights(module)

    def embed(self, tokens: torch.Tensor):
        return self.embedding(tokens)

    def enable_metal_trunk(self, max_length: int | None = None) -> None:
        super().enable_metal_trunk(max_length)
        self.embedding.grid.fused = metal_grid_supported(self.config)

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)
