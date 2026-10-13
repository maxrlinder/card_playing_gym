"""Rummy actor and oracle critic on the shared trunk.

Input. A token is its event fields (one packed lookup, as in plump) plus a
summary of the viewer's ledger computed by :class:`CardGrid`:

* each card type starts as the embedding of its ledger code (where both
  copies are, moved, layable) plus a rank embedding (points break the rank
  symmetry the rules otherwise have);
* grid blocks on the 4 x 13 suit-by-rank grid mix a circular convolution
  along the rank cycle (runs are cyclic) with each rank's mean over suits
  (groups). Both are equivariant to suit permutations and rank rotations, the
  rules' own symmetries, so meld structure is learned once for every suit and
  every rank;
* a per-card linear map flattens the grid into the token embedding, which
  keeps card identity (and therefore injectivity) rather than pooling it away.

The grid runs once per token, before the trunk. It is the token's input, so a
cached key never goes stale: each position keeps the ledger of its own moment.

Output. Both policy heads score the engine's candidate list. A candidate's
embedding sums its kind, its moved cards and (for a lay-off) the target
meld's cards, each card represented by its grid features at the decision
point plus an identity row. The readout hidden state is added and an MLP
scores the result (additive attention), so the head sees card-level detail
the pooled token summary may blur.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cardgym.games.base import CriticModel, PolicyModel
from cardgym.nn.model import value_mlp
from cardgym.nn.trunk import RMSNorm, TrunkConfig, ValueEmbeddingMode, init_weights

from .cards import CARD_RANKS, NO_CARD, NUM_CARDS, NUM_RANKS, NUM_SUITS
from .engine import MAX_PLAYERS, NUM_ACTION_KINDS
from .selfplay import MAX_MELD_CARDS
from .tokens import FIELD_VOCAB, LEDGER_START, LEDGER_VOCAB, NUM_FIELDS

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
    per_layer_dim: int = 32
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


class GridBlock(nn.Module):
    """Residual mix of rank neighbours (circular) and same-rank suits on the card grid."""

    def __init__(self, width: int, hidden: int, kernel: int) -> None:
        super().__init__()
        self.norm = RMSNorm(width)
        self.along_rank = nn.Conv1d(width, hidden, kernel, padding=kernel // 2, padding_mode="circular")
        self.across_suits = nn.Linear(width, hidden, bias=False)
        self.out = nn.Linear(hidden, width)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:  # [N, suits, ranks, width]
        n, suits, ranks, width = grid.shape
        y = self.norm(grid)
        along = self.along_rank(y.reshape(n * suits, ranks, width).transpose(1, 2))
        along = along.transpose(1, 2).reshape(n, suits, ranks, -1)
        across = self.across_suits(y.mean(dim=1, keepdim=True))
        return grid + self.out(F.gelu(along + across))


class CardGrid(nn.Module):
    """Ledger codes [..., 52] -> per-card features [..., 52, width] and a token summary [..., d]."""

    def __init__(self, config: RummyModelConfig) -> None:
        super().__init__()
        width = config.ledger_width
        self.width = width
        self.code = nn.Embedding(LEDGER_VOCAB, width)
        self.rank = nn.Embedding(NUM_RANKS, width)
        self.blocks = nn.ModuleList(
            GridBlock(width, config.ledger_hidden, config.ledger_kernel) for _ in range(config.ledger_layers)
        )
        self.norm = RMSNorm(width)
        self.summary = nn.Linear(NUM_CARDS * width, config.d_model, bias=False)
        for module in (self.code, self.rank, *self.blocks):
            module.apply(init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.out.weight.mul_(1.0 / max(len(self.blocks), 1) ** 0.5)
        # Comparable scale to the sum of the event-field rows.
        nn.init.normal_(self.summary.weight, std=0.02 * (NUM_FIELDS / (NUM_CARDS * width)) ** 0.5)
        self.register_buffer("card_ranks", torch.as_tensor(CARD_RANKS, dtype=torch.long), persistent=False)

    def cards(self, ledger: torch.Tensor) -> torch.Tensor:
        lead = ledger.shape[:-1]
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
    def __init__(self, config: RummyModelConfig, ple_width: int, value_width: int) -> None:
        super().__init__()
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
        _, summary = self.grid(tokens[..., LEDGER_START:])
        return (
            _bag(ids, self.fields.weight) + summary,
            None if self.ple_fields is None else _bag(ids, self.ple_fields.weight),
            None if self.value_fields is None else _bag(ids, self.value_fields.weight),
        )


def _card_sum(table: torch.Tensor, cards: torch.Tensor) -> torch.Tensor:
    """Sum the rows of ``table`` [N, 52, D] named by ``cards`` [N, K, C]; NO_CARD adds nothing."""

    hot = torch.zeros(*cards.shape[:2], NUM_CARDS + 1, device=table.device, dtype=table.dtype)
    hot.scatter_add_(2, cards, torch.ones_like(cards, dtype=table.dtype))
    return hot[..., :NUM_CARDS] @ table


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
        self.norm = nn.LayerNorm(dim)
        self.hidden = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 1)
        self.apply(init_weights)

    def forward(
        self,
        hidden: torch.Tensor,
        grid: torch.Tensor,
        kind: torch.Tensor,
        cards: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        cards = cards.long()
        target = target.long()
        moved_rows = self.card(grid) + self.card_id.weight.to(grid.dtype)
        target_rows = self.target(grid) + self.target_id.weight.to(grid.dtype)
        candidate = (
            _card_sum(moved_rows, cards)
            + _card_sum(target_rows, target)
            + self.kind(kind.long())
            + self.size((cards != NO_CARD).sum(-1))
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
        self.embedding = RummyEmbedding(config, trunk.ple_width, trunk.value_width)
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

    def policy_logits(self, hidden, head, features=None, tokens=None):
        if features is None or tokens is None:
            raise ValueError("Rummy policy heads score candidate lists at a readout token.")
        if head not in self.policy_heads:
            raise ValueError(f"Unknown rummy policy head {head!r}.")
        grid = self.embedding.grid.cards(tokens[..., LEDGER_START:])
        return self.scorer(hidden, grid, features["kind"], features["cards"], features["target"])

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)


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

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)
