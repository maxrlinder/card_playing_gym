"""Continuous self-play: a fixed pool of games in flight, refilled as games end.

The batch collector (:class:`~cardgym.rl.selfplay.SelfPlayCollector`) deals
one update's games together and decodes until the last one ends, so the
number of waves follows the longest game and the live batch thins out behind
it. Here every lane keeps ``in_flight`` games running at all times: a game
that ends is replaced in the same wave, and :meth:`ContinuousCollector.collect`
returns as soon as each lane has finished its quota. Games still running carry
over, with their cache rows, into the next call, which decodes them with the
new weights. A collection therefore costs about ``quota * mean length /
in_flight`` waves, not the longest game.

Off-policy by construction. A carried-over game's later decisions are sampled
by newer weights reading keys and values that older weights wrote. Each
decision records the exact probabilities it was sampled from and PPO's ratio
is taken against those, so the update stays correct; ``policy_lag`` reports
how many versions old the returned decisions are.

Cohorts. With ``cohorts = 2`` the slots alternate between two groups that
take turns: while the device decodes one group's wave, the host samples and
steps the other's games, so a collector's host work and device work overlap
instead of alternating. Nothing is in flight between collections.

Rows. Each game owns one cache row per seat. A wave decodes only the rows that
received tokens (in Rummy, just the acting seat's), gathered by index, so the
forward batch is the number of live games rather than of observers. The row
count is bucketed to a multiple of ``ROW_BUCKET`` (MPS compiles a graph per
shape); padding entries write to a scratch row past every game's rows.
"""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from cardgym.core.devices import DTYPES, Fence, autocast_context, release_device_memory, to_device

from .selfplay import (
    CollectionStats,
    SelfPlayGame,
    Trajectory,
    _Live,
    _round_up,
    launch_policies,
    sample_decisions,
)


@dataclass(frozen=True)
class Lane:
    """One kind of game kept running.

    ``deal(rng)`` returns a fresh game, or None once a finite source is spent.
    With ``quota = None`` the lane runs until its source is spent and every
    game has ended (evaluation); otherwise each collection returns exactly
    ``quota`` finished games from it.
    """

    name: str
    players: int  # cache rows per game
    in_flight: int
    quota: int | None
    deal: Callable[[random.Random], SelfPlayGame | None] = field(compare=False)


def split_lanes(lanes: list[Lane], index: int, count: int) -> list[Lane]:
    """Worker ``index`` of ``count``'s share of each lane: its part of the quota and of the games in flight.

    Shares differ by at most one game; a lane with nothing for this worker is left out.
    """

    shares = []
    for lane in lanes:
        if lane.quota is None:
            raise ValueError("Only lanes with a quota can be split between workers.")
        quota = lane.quota // count + (index < lane.quota % count)
        in_flight = lane.in_flight // count + (index < lane.in_flight % count)
        if quota > 0:
            shares.append(replace(lane, quota=quota, in_flight=max(1, in_flight)))
    return shares


@dataclass
class _Slot:
    lane: int
    base: int  # first cache row
    cohort: int = 0
    state: _Live | None = None
    queued: list[tuple[int, np.ndarray]] = field(default_factory=list)  # (seat, block) not yet decoded


@dataclass
class _Wave:
    """One cohort's decode in flight: its policy reads on their way to the host."""

    launched: list = field(default_factory=list)
    fence: Fence | None = None
    # Host arrays that in-flight non-blocking uploads still read from.
    keep: list[np.ndarray] = field(default_factory=list)


class ContinuousCollector:
    ROW_BUCKET = 64
    SPAN_BUCKET = 32

    def __init__(
        self,
        model,
        *,
        device: torch.device,
        precision: str,
        kv_dtype: str,
        max_length: int,
        cohorts: int = 1,
    ) -> None:
        self.model = model
        self.device = device
        self.precision = precision
        self.kv_dtype = DTYPES[kv_dtype]
        self.cohorts = max(1, cohorts)
        # One spare position past the longest stream absorbs ragged padding.
        self.cache_length = max_length + 1
        self.stats = CollectionStats()
        self._layout: tuple | None = None
        self._slots: list[_Slot] = []
        self._finished: dict[str, deque[Trajectory]] = {}
        self._cache = None
        self._scratch = 0
        self._waves = [_Wave() for _ in range(self.cohorts)]
        self._width: int | None = None

    def reset(self) -> None:
        """Drop every game in flight, every banked result, and the cache."""

        self._layout = None
        self._slots = []
        self._finished = {}
        self._cache = None
        self._waves = [_Wave() for _ in range(self.cohorts)]
        release_device_memory(self.device)

    def release(self) -> None:
        """Nothing to free between updates: the pool holds the games in flight."""

    @property
    def in_flight(self) -> int:
        return sum(slot.state is not None for slot in self._slots)

    # no_grad rather than inference_mode: autocast does not cache its bf16
    # weight casts under inference_mode (measured 3 casts per linear per call).
    @torch.no_grad()
    def collect(self, lanes: list[Lane], rng: random.Random, *, version: int = 0) -> list[Trajectory]:
        started = time.perf_counter()
        self.stats = CollectionStats()
        self._configure(lanes)
        for slot in self._slots:
            if slot.state is None:
                self._deal(slot, lanes, rng)
        # One autocast context for the whole collection: autocast caches its
        # bf16 copies of the weights until the outermost context exits, and on
        # MPS each cast is a ~0.2 ms dispatch, paid per weight per wave otherwise.
        with autocast_context(self.device, self.precision):
            pending = [False] * self.cohorts
            while True:
                for cohort in range(self.cohorts):
                    if pending[cohort]:
                        self._receive(cohort)
                        self._act(cohort, lanes, rng, version)
                        pending[cohort] = False
                    if not self._satisfied(lanes) and self.in_flight:
                        pending[cohort] = self._launch(cohort)
                if not any(pending):
                    break
        results: list[Trajectory] = []
        for lane in lanes:
            bank = self._finished[lane.name]
            count = len(bank) if lane.quota is None else min(lane.quota, len(bank))
            results.extend(bank.popleft() for _ in range(count))
        decisions = [record.version for trajectory in results for record in trajectory.decisions]
        self.stats.games = len(results)
        self.stats.in_flight = self.in_flight
        self.stats.policy_lag = float(version - np.mean(decisions)) if decisions else 0.0
        self.stats.collect_sec = time.perf_counter() - started
        return results

    # ------------------------------------------------------------------

    def _configure(self, lanes: list[Lane]) -> None:
        layout = tuple((lane.name, lane.players, lane.in_flight) for lane in lanes)
        if layout == self._layout:
            return
        self.reset()
        self._layout = layout
        row = 0
        for index, lane in enumerate(lanes):
            self._finished[lane.name] = deque()
            for _ in range(lane.in_flight):
                self._slots.append(_Slot(lane=index, base=row, cohort=len(self._slots) % self.cohorts))
                row += lane.players
        self._scratch = row
        self._cache = self.model.new_cache(row + 1, self.cache_length, self.kv_dtype)

    def _satisfied(self, lanes: list[Lane]) -> bool:
        for index, lane in enumerate(lanes):
            if lane.quota is None:
                if any(slot.state is not None for slot in self._slots if slot.lane == index):
                    return False
            elif len(self._finished[lane.name]) < lane.quota:
                return False
        return True

    def _deal(self, slot: _Slot, lanes: list[Lane], rng: random.Random) -> None:
        """Fill an empty slot; games that end before any decision finish at once.

        A lane whose finished games already cover its quota deals nothing:
        a collection runs until the slowest lane meets its quota, and a lane
        with shorter games would otherwise bank surplus games every time,
        each returned later and older (policy lag grew to 5+ versions).
        """

        lane = lanes[slot.lane]
        if lane.quota is not None and len(self._finished[lane.name]) >= lane.quota:
            return
        while True:
            game = lane.deal(rng)
            if game is None:
                return
            state = _Live(
                game=game,
                base=slot.base,
                lengths=np.zeros(game.num_players, dtype=np.int64),
                blocks=[[] for _ in range(game.num_players)],
            )
            queued = self._record(state, game.prefix_tokens())
            if game.done():
                self._finished[lane.name].append(self._finish(state))
                continue
            slot.state = state
            slot.queued = queued
            self._expect_readout(state, queued)
            return

    def _record(self, state: _Live, blocks: list[np.ndarray]) -> list[tuple[int, np.ndarray]]:
        queued = []
        for seat, block in enumerate(blocks):
            if len(block):
                state.blocks[seat].append(block)
                queued.append((seat, block))
                self._width = block.shape[1]
        return queued

    @staticmethod
    def _expect_readout(state: _Live, queued: list[tuple[int, np.ndarray]]) -> None:
        state.pending = state.game.decision()
        seat = state.pending.seat
        blocks = [block for queued_seat, block in queued if queued_seat == seat]
        if not blocks:
            raise AssertionError("The next actor's stream received no readout token.")
        state.readout = blocks[-1][-1]

    def _finish(self, state: _Live) -> Trajectory:
        width = self._width or 0
        streams = [
            np.concatenate(blocks, axis=0) if blocks else np.zeros((0, width), dtype=np.int16)
            for blocks in state.blocks
        ]
        return Trajectory(
            game=state.game,
            streams=streams,
            decisions=state.decisions,
            returns=np.asarray(state.game.returns(), dtype=np.float64),
        )

    def _upload(self, array: np.ndarray, cohort: int) -> torch.Tensor:
        self._waves[cohort].keep.append(array)
        return to_device(array, self.device)

    def _launch(self, cohort: int) -> bool:
        """Decode the cohort's queued blocks; queue the policy read of its pending decisions.

        Returns whether anything was launched (the cohort has live games).
        """

        started = time.perf_counter()
        entries: list[tuple[int, int, np.ndarray]] = []  # (row, position, block)
        readouts: list[tuple[int, _Live]] = []
        for slot in self._slots:
            state = slot.state
            if state is None or slot.cohort != cohort:
                continue
            actor = state.pending.seat
            for seat, block in slot.queued:
                if seat == actor:
                    readouts.append((len(entries), state))
                entries.append((slot.base + seat, int(state.lengths[seat]), block))
                state.lengths[seat] += len(block)
            if int(state.lengths.max()) >= self.cache_length:
                raise RuntimeError("A stream outgrew the model's maximum sequence length.")
            slot.queued = []
        if not entries:
            return False
        upload = lambda array: self._upload(array, cohort)  # noqa: E731
        run = max(len(block) for _, _, block in entries)
        count = len(entries)
        rows = _round_up(count, self.ROW_BUCKET)
        width = entries[0][2].shape[1]
        tokens = np.zeros((rows, run, width), dtype=np.int64)
        cache_rows = np.full(rows, self._scratch, dtype=np.int64)
        positions = np.zeros(rows, dtype=np.int64)
        offsets = np.zeros(rows, dtype=np.int64)
        for index, (row, position, block) in enumerate(entries):
            tokens[index, : len(block)] = block
            tokens[index, len(block) :] = block[-1]  # padding repeats the last token
            cache_rows[index] = row
            positions[index] = position
            offsets[index] = len(block) - 1
        span = _round_up(int(positions.max()) + run, self.SPAN_BUCKET)
        with autocast_context(self.device, self.precision):
            hidden = self.model.forward_ragged(
                upload(tokens),
                upload(positions),
                self._cache,
                upload(offsets),
                span=span,
                rows=upload(cache_rows),
            )
        context = self.model.pop_context()
        index = upload(np.asarray([entry for entry, _ in readouts], dtype=np.int64))
        if context is not None:
            context = context[index, upload(offsets[[entry for entry, _ in readouts]])]
        wave = self._waves[cohort]
        wave.launched = launch_policies(
            self.model, self.device, self.precision, upload,
            hidden[index], [state for _, state in readouts], context,
        )
        wave.fence = Fence(self.device)
        self.stats.waves += 1
        self.stats.forward_rows += rows * run
        self.stats.padded_rows += rows * run - sum(len(block) for _, _, block in entries)
        self.stats.peak_rows = max(self.stats.peak_rows, count)
        self.stats.forward_sec += time.perf_counter() - started
        return True

    def _receive(self, cohort: int) -> None:
        started = time.perf_counter()
        wave = self._waves[cohort]
        if wave.fence is not None:
            wave.fence.wait()
        self.stats.wait_sec += time.perf_counter() - started
        for states, logits in wave.launched:
            values = logits.numpy()
            for row, state in enumerate(states):
                state.logits = values[row]
        self._waves[cohort] = _Wave()

    def _act(self, cohort: int, lanes: list[Lane], rng: random.Random, version: int) -> None:
        """Sample and apply the cohort's pending decisions; retire and replace finished games."""

        t0 = time.perf_counter()
        live = [slot for slot in self._slots if slot.state is not None and slot.cohort == cohort]
        self.stats.decisions += sample_decisions([slot.state for slot in live], version)
        t1 = time.perf_counter()
        for slot in live:
            state = slot.state
            queued = self._record(state, state.game.step(state.decisions[-1].action))
            if state.game.done():
                self._finished[lanes[slot.lane].name].append(self._finish(state))
                slot.state = None
                self._deal(slot, lanes, rng)
            else:
                slot.queued = queued
                self._expect_readout(state, queued)
        self.stats.sample_sec += t1 - t0
        self.stats.step_sec += time.perf_counter() - t1
