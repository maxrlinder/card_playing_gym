"""Piquet deals (or Rubicon parties) as self-play games for the shared collectors.

Every event reaches both seats' streams as it happens; the seat about to
decide then gets its decision token last, the readout its policy reads. All
three heads are fixed: 33 exchange actions (a card, or stop), 2 declare
actions, 32 cards to play, masked to the legal ones. Besides the streams the
game records, per token, the deal it belongs to, how many categories,
tricks, and cards of that deal were settled or played, and the true location
of every card the viewer cannot place: what training needs for its labels.

A partie gives each deal a fresh context: :meth:`PiquetSelfPlay.context_starts`
marks where the next deal's tokens begin, the collectors restart the KV
cache there, and training splits each seat's stream the same way. The
score fields of every token carry what the earlier deals decided.
"""

from __future__ import annotations

import random

import numpy as np

from cardgym.rl.selfplay import Decision, SelfPlayGame

from .engine import Event, PiquetEngine
from .tokens import WIDTH, ObservationWriter

EMPTY = np.zeros((0, WIDTH), dtype=np.int16)
MODES = ("deal", "rubicon")
RUBICON_DEALS = 6
# Decisions in a typical deal: about 7 exchange choices, 4 declarations, 18 cards.
DECISIONS_PER_DEAL = 30


class PiquetSelfPlay(SelfPlayGame):
    """One deal (``mode = "deal"``) or a six-deal Rubicon partie, both seats played by the policy."""

    def __init__(self, seed: int, *, mode: str = "deal", deck: list[int] | None = None, dealer: int | None = None):
        if mode not in MODES:
            raise ValueError(f"Unknown piquet mode {mode!r}; expected one of {MODES}.")
        self.num_players = 2
        self.mode = mode
        self.seed = seed
        self.rng = random.Random(seed ^ 0x5EED)
        self.writer = ObservationWriter()
        self.pending: list[list[np.ndarray]] = [[], []]
        # Per seat, per token: (deal, categories settled, tricks finished, cards played) and the belief truth.
        self.meta: list[list[tuple[int, int, int, int]]] = [[], []]
        self.truth: list[list[np.ndarray]] = [[], []]
        self._decision: Decision | None = None
        # Per seat: the deal of its last token, and where the next deal's tokens
        # begin in the block being gathered (and in the one the last step returned).
        self._deal_of = [0, 0]
        self._starting: list[int | None] = [None, None]
        self._starts: list[int | None] | None = None
        rubicon = mode == "rubicon"
        self.engine = PiquetEngine(
            seed, deals=RUBICON_DEALS if rubicon else 1, rubicon=rubicon, dealer=dealer, deck=deck,
            listener=self._on_event,
        )
        self._prefix = self._flush()

    def _push(self, engine: PiquetEngine, seat: int, token: np.ndarray) -> None:
        if engine.deal_index != self._deal_of[seat]:
            self._deal_of[seat] = engine.deal_index
            self._starting[seat] = len(self.pending[seat])
        self.pending[seat].append(token)
        played = 2 * engine.trick_index + len(engine.trick)
        self.meta[seat].append((engine.deal_index, engine.category, engine.trick_index, played))
        self.truth[seat].append(self.writer.truth(engine, seat))

    def _on_event(self, engine: PiquetEngine, event: Event) -> None:
        rows = self.writer.event_rows(engine, event)
        if rows is not None:
            for seat in (0, 1):
                self._push(engine, seat, rows[seat])

    def _flush(self) -> list[np.ndarray]:
        engine = self.engine
        if not engine.done:
            self._push(engine, engine.current, self.writer.decision_row(engine, engine.current))
        blocks = [np.stack(rows) if rows else EMPTY for rows in self.pending]
        self.pending = [[], []]
        self._starts = None if self._starting == [None, None] else self._starting
        self._starting = [None, None]
        return blocks

    def context_starts(self) -> list[int | None] | None:
        return self._starts

    def lifetime(self) -> int:
        engine = self.engine
        return (engine.deals - engine.deal_index) * DECISIONS_PER_DEAL - engine.trick_index

    def prefix_tokens(self) -> list[np.ndarray]:
        return self._prefix

    def decision(self) -> Decision:
        if self._decision is None:
            engine = self.engine
            self._decision = Decision(seat=engine.current, head=engine.head, legal=engine.legal())
        return self._decision

    def step(self, action: int) -> list[np.ndarray]:
        self.engine.apply(action)
        self._decision = None
        return self._flush()

    def done(self) -> bool:
        return self.engine.done

    def returns(self) -> np.ndarray:
        return self.engine.returns()

