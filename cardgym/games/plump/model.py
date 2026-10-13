"""Plump actor and oracle critic on the shared trunk.

Input: each token is the sum of one embedding row per present categorical
field (15 fields; a card is rank + suit, never an exact-card row), one row
for the token's pattern of absent fields, and, on TRICK_WIN rows, the rank +
suit rows of every card the observer still holds. The same field lookup,
from separate tables, supplies the packed per-layer (PLE) and attention-value
conditioning.

Actor heads:

* ``bid``: logits over 0..max_hand_size.
* ``play``: 52 card logits, each ``h . (rank[r] + suit[s]) + bias[r] + bias[s]``
  so rank behaviour is shared across suits and vice versa.
* per-seat value probe (zero-sum, observer-relative): a measurement of what
  observer information predicts; it never feeds advantages.
* auxiliary beliefs: opponent suit presence, rank boundaries, final trick
  counts, next trick winner (packed linear), and bid hit (an MLP: "won
  exactly the bid" is a bump, not a threshold).

The critic sees the oracle stream (every hand) and predicts per-seat values
on the absolute seat axis, plus final tricks, next trick winner, and each
seat's future play order.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cardgym.games.base import CriticModel, PolicyModel
from cardgym.nn.model import value_mlp
from cardgym.nn.trunk import TrunkConfig, ValueEmbeddingMode, init_weights

from .tokens import (
    BASE_WIDTH,
    NA_PATTERN_MASKS,
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    SLOT_CARD,
    SLOT_NA_PATTERN,
    SLOT_RANK,
    SLOT_REMAINING_HAND,
    SLOT_SUIT,
    SLOT_TYPE,
    TOKEN_TRICK_WIN,
    Schema,
)

# Base fields the model embeds: everything but the exact card id.
FIELDS = tuple(slot for slot in range(BASE_WIDTH) if slot != SLOT_CARD)


@dataclass(frozen=True)
class PlumpModelConfig:
    """The [model] section. Defaults are the rl-v6 architecture (8.6M params)."""

    max_players: int = 5
    max_hand_size: int = 10
    d_model: int = 240
    n_layers: int = 16
    n_heads: int = 10
    n_kv_heads: int = 2
    d_ff: int = 720
    per_layer_dim: int = 64
    value_embeddings: ValueEmbeddingMode = "alternating"

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

    def schema(self) -> Schema:
        return Schema(max_players=self.max_players, max_hand_size=self.max_hand_size)

    def validate(self) -> None:
        self.trunk().validate()
        if not 3 <= self.max_players <= 7 or self.max_hand_size < 1:
            raise ValueError("[model] max_players must be 3..7 and max_hand_size positive.")


def _bag(indices: torch.Tensor, weight: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Sum of selected rows per bag without a [..., bag, dim] temporary."""

    flat = indices.reshape(-1, indices.shape[-1])
    summed = F.embedding_bag(
        flat, weight, mode="sum", per_sample_weights=valid.reshape_as(flat).to(weight.dtype)
    )
    return summed.view(*indices.shape[:-1], weight.shape[-1])


class PlumpEmbedding(nn.Module):
    def __init__(self, schema: Schema, d_model: int, ple_width: int, value_width: int) -> None:
        super().__init__()
        sizes = [schema.base_vocab_sizes[slot] for slot in FIELDS]
        offsets = torch.tensor([sum(sizes[:i]) for i in range(len(sizes))], dtype=torch.long)
        na_ids = torch.tensor(
            [schema.base_na_ids[slot] if schema.base_na_ids[slot] is not None else -1 for slot in FIELDS],
            dtype=torch.long,
        )
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("na_ids", na_ids, persistent=False)
        self.register_buffer("field_slots", torch.tensor(FIELDS, dtype=torch.long), persistent=False)
        self.rank_offset = int(offsets[FIELDS.index(SLOT_RANK)])
        self.suit_offset = int(offsets[FIELDS.index(SLOT_SUIT)])
        rows = int(sum(sizes))
        patterns = len(NA_PATTERN_MASKS)
        self.fields = nn.Embedding(rows, d_model)
        self.na_pattern = nn.Embedding(patterns, d_model)
        init_weights(self.fields)
        init_weights(self.na_pattern)
        self.ple_fields = nn.Embedding(rows, ple_width) if ple_width else None
        self.ple_na_pattern = nn.Embedding(patterns, ple_width) if ple_width else None
        self.value_fields = nn.Embedding(rows, value_width) if value_width else None
        self.value_na_pattern = nn.Embedding(patterns, value_width) if value_width else None
        with torch.no_grad():
            self.na_pattern.weight[0].zero_()  # padding rows contribute nothing
            # Conditioning starts at zero: enabling it leaves the function unchanged.
            for table in (self.ple_fields, self.ple_na_pattern, self.value_fields, self.value_na_pattern):
                if table is not None:
                    table.weight.zero_()

    def forward(self, tokens: torch.Tensor):
        fields = tokens[..., self.field_slots]
        ids = fields + self.offsets
        valid = (self.na_ids < 0) | (fields != self.na_ids)
        pattern = tokens[..., SLOT_NA_PATTERN]
        # Resolve the TRICK_WIN rows to explicit indices once and reuse them:
        # indexing twice with one boolean mask has been seen to disagree with
        # itself on MPS.
        trick_win = (tokens[..., SLOT_TYPE] == TOKEN_TRICK_WIN).nonzero(as_tuple=True)
        remaining = tokens[..., SLOT_REMAINING_HAND:SLOT_NA_PATTERN][trick_win]
        hand_rows = hand_valid = None
        if remaining.numel():
            card_valid = remaining < NUM_CARDS
            safe = remaining.clamp_max(NUM_CARDS - 1)
            ranks = safe.remainder(NUM_RANKS) + self.rank_offset
            suits = safe.div(NUM_RANKS, rounding_mode="floor") + self.suit_offset
            hand_rows = torch.stack((ranks, suits), dim=-1).flatten(-2)
            hand_valid = card_valid.unsqueeze(-1).expand(*card_valid.shape, 2).flatten(-2)

        def lookup(table: nn.Embedding | None, na_table: nn.Embedding | None):
            if table is None:
                return None
            out = _bag(ids, table.weight, valid) + na_table(pattern)
            if hand_rows is not None:
                out = out.clone()
                out[trick_win] = out[trick_win] + _bag(hand_rows, table.weight, hand_valid)
            return out

        return (
            lookup(self.fields, self.na_pattern),
            lookup(self.ple_fields, self.ple_na_pattern),
            lookup(self.value_fields, self.value_na_pattern),
        )


class PlumpActor(PolicyModel):
    policy_heads = ("bid", "play")
    AUXILIARY = ("suit", "rank_boundary", "trick", "next_winner")

    def __init__(self, config: PlumpModelConfig) -> None:
        super().__init__(config.trunk())
        self.config = config
        self.schema = schema = config.schema()
        trunk = self.trunk_config
        d = config.d_model
        self.embedding = PlumpEmbedding(schema, d, trunk.ple_width, trunk.value_width)
        self.bid_head = nn.Linear(d, schema.bid_count)
        self.card_rank = nn.Embedding(NUM_RANKS, d)
        self.card_suit = nn.Embedding(NUM_SUITS, d)
        self.card_rank_bias = nn.Parameter(torch.zeros(NUM_RANKS))
        self.card_suit_bias = nn.Parameter(torch.zeros(NUM_SUITS))
        self.value_head = value_mlp(d, config.max_players)
        self.suit_head = nn.Linear(d, schema.opponents * NUM_SUITS)
        self.rank_boundary_head = nn.Linear(d, (schema.opponents + 1) * NUM_SUITS * 2)
        self.trick_head = nn.Linear(d, config.max_players * schema.bid_count)
        self.next_winner_head = nn.Linear(d, config.max_players)
        self.bid_hit_head = value_mlp(d, config.max_players)
        for module in (
            self.bid_head, self.card_rank, self.card_suit, self.suit_head,
            self.rank_boundary_head, self.trick_head, self.next_winner_head,
        ):
            init_weights(module)
        cards = torch.arange(NUM_CARDS)
        self.register_buffer("card_ranks", cards % NUM_RANKS, persistent=False)
        self.register_buffer("card_suits", cards // NUM_RANKS, persistent=False)

    def embed(self, tokens: torch.Tensor):
        return self.embedding(tokens)

    def card_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = self.card_rank(self.card_ranks) + self.card_suit(self.card_suits)
        bias = self.card_rank_bias[self.card_ranks] + self.card_suit_bias[self.card_suits]
        return F.linear(hidden, weight, bias)

    def policy_logits(self, hidden, head, features=None, tokens=None, context=None):
        if head == "bid":
            return self.bid_head(hidden)
        if head == "play":
            return self.card_logits(hidden)
        raise ValueError(f"Unknown plump policy head {head!r}.")

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)

    def auxiliary_logits(self, hidden: torch.Tensor, heads: Collection[str]) -> dict[str, torch.Tensor]:
        """Selected linear beliefs from one packed projection (one matmul)."""

        modules = {
            "suit": self.suit_head,
            "rank_boundary": self.rank_boundary_head,
            "trick": self.trick_head,
            "next_winner": self.next_winner_head,
        }
        ordered = [name for name in self.AUXILIARY if name in heads]
        if not ordered:
            return {}
        packed = F.linear(
            hidden,
            torch.cat([modules[name].weight for name in ordered]),
            torch.cat([modules[name].bias for name in ordered]),
        )
        return dict(zip(ordered, packed.split([modules[n].out_features for n in ordered], dim=-1)))


class PlumpCritic(CriticModel):
    AUXILIARY = ("trick", "card_order", "next_winner")

    def __init__(self, config: PlumpModelConfig) -> None:
        super().__init__(config.trunk())
        self.config = config
        self.schema = schema = config.schema()
        trunk = self.trunk_config
        d = config.d_model
        self.embedding = PlumpEmbedding(schema, d, trunk.ple_width, trunk.value_width)
        self.value_head = value_mlp(d, config.max_players)
        self.trick_head = nn.Linear(d, config.max_players * schema.bid_count)
        # One contextual score per exact card. Hands are disjoint, so the
        # trainer normalizes scores within each owner's remaining cards.
        self.card_order_head = nn.Linear(d, NUM_CARDS)
        self.next_winner_head = nn.Linear(d, config.max_players)
        for module in (self.trick_head, self.card_order_head, self.next_winner_head):
            init_weights(module)

    def embed(self, tokens: torch.Tensor):
        return self.embedding(tokens)

    def player_values(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden)

    def auxiliary_logits(self, hidden: torch.Tensor, heads: Collection[str]) -> dict[str, torch.Tensor]:
        modules = {"trick": self.trick_head, "card_order": self.card_order_head, "next_winner": self.next_winner_head}
        ordered = [name for name in self.AUXILIARY if name in heads]
        if not ordered:
            return {}
        packed = F.linear(
            hidden,
            torch.cat([modules[name].weight for name in ordered]),
            torch.cat([modules[name].bias for name in ordered]),
        )
        return dict(zip(ordered, packed.split([modules[n].out_features for n in ordered], dim=-1)))
