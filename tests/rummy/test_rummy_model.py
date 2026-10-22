"""Rummy model pieces against their direct definitions (CPU, fp32)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cardgym.games.rummy.cards import NO_CARD, NUM_CARDS
from cardgym.games.rummy.engine import NUM_ACTION_KINDS
from cardgym.games.rummy.model import CandidateScorer
from cardgym.games.rummy.selfplay import MAX_MELD_CARDS


def _copies(cards: torch.Tensor) -> torch.Tensor:
    hot = torch.zeros(*cards.shape[:2], NUM_CARDS + 1)
    hot.scatter_add_(2, cards.long(), torch.ones_like(cards, dtype=torch.float32))
    return hot[..., :NUM_CARDS]


def _direct(scorer: CandidateScorer, hidden, grid, kind, cards, target, taken):
    """Each slot's 52 card features projected and summed, one slot at a time."""

    moved, aimed = _copies(cards), _copies(target)
    candidate = (
        scorer.card(moved @ grid) + moved @ scorer.card_id.weight
        + scorer.target(aimed @ grid) + aimed @ scorer.target_id.weight
        + scorer.kind(kind.long()) + scorer.size((cards != NO_CARD).sum(-1))
    )
    if taken is not None:
        taken = taken.float()
        candidate = candidate + scorer.taken(taken @ grid) + taken @ scorer.taken_id.weight
    mixed = scorer.norm(scorer.query(hidden)[:, None, :] + candidate)
    return scorer.out(F.gelu(scorer.hidden(mixed))).squeeze(-1)


@pytest.mark.parametrize("draw", [False, True])
def test_candidate_scorer_matches_its_definition(draw):
    torch.manual_seed(0)
    rows, width, d, card_width = 7, 9, 48, 32
    scorer = CandidateScorer(d, card_width, 16)
    hidden = torch.randn(rows, d, requires_grad=True)
    grid = torch.randn(rows, NUM_CARDS, card_width, requires_grad=True)
    kind = torch.randint(0, NUM_ACTION_KINDS, (rows, width), dtype=torch.int8)
    cards = torch.randint(0, NUM_CARDS + 1, (rows, width, MAX_MELD_CARDS), dtype=torch.int8)
    target = torch.randint(0, NUM_CARDS + 1, (rows, width, MAX_MELD_CARDS), dtype=torch.int8)
    cards[..., 4:] = NO_CARD  # mostly short lists, as in play
    taken = torch.randint(0, 3, (rows, width, NUM_CARDS), dtype=torch.int8) if draw else None
    fused = scorer(hidden, grid, kind, cards, target, taken)
    direct = _direct(scorer, hidden, grid, kind, cards, target, taken)
    torch.testing.assert_close(fused, direct, rtol=1e-5, atol=1e-5)
    weights = torch.randn_like(direct)
    inputs = [hidden, grid, *scorer.parameters()]
    got = torch.autograd.grad((fused * weights).sum(), inputs, allow_unused=True)
    want = torch.autograd.grad((direct * weights).sum(), inputs, allow_unused=True)
    for a, b in zip(got, want):
        if b is None:
            assert a is None or not a.abs().any()
        else:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
