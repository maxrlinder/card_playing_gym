"""Continuously batched, KV-cached self-play for any multi-observer game.

Every game in a schedule is decoded by one merged loop. Each observer of each
game owns one cache row, and every row carries its own length, so games of
different shapes, at different stages, share one ragged forward per wave:

    wave:  every live game makes one decision (actor row's readout)
           -> sample from the masked policy, step the game
           -> append each observer's new tokens (padded to the wave's
              longest run) and read the next actor's hidden state

Live rows must stay a dense prefix of the pool: the ragged attention reads
rows 0..B-1 as a zero-copy slice (a per-row gather measured up to 1.65x
slower). Games are therefore dealt longest-lived first so retirement clears
rows from the tail; a game that finishes out of turn triggers a compaction
that slides the survivors down, which keeps the invariant exact.

Cohorts overlap host and device. With ``cohorts = 2`` the games are split
into two groups with their own caches, and the loop alternates: while the
device decodes one cohort's wave, the host samples and steps the other's
games. Every upload and every logits download is non-blocking and the host
waits only on the fence of the cohort it is about to step; on MPS a blocking
copy would wait for all queued work, including the other cohort's.

Decode shapes are bucketed (live rows to a multiple of 64, the attention span
to a multiple of 32, policy reads to fixed decision and candidate counts).
MPS compiles a graph per tensor shape; exact shapes change every wave and
measured 24-27 ms per forward against 15.5 ms bucketed. Padding rows write
only to cache rows past the live prefix and their outputs are dropped.

A game plugs in through :class:`SelfPlayGame`. It owns all rules and all
tokenization; the collector only moves token blocks and samples actions.
"""

from __future__ import annotations

import abc
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from cardgym.core.devices import DTYPES, Fence, autocast_context, release_device_memory, to_device, upload_packed

from .ppo import masked_probabilities, sample_rows


@dataclass
class Decision:
    """What the acting seat must choose now."""

    seat: int
    head: str
    legal: np.ndarray  # bool [actions]; actions index the head's output
    # Pointer heads score a variable candidate list; each entry is an array
    # whose leading axis is the candidate axis. Fixed heads leave this None.
    features: dict[str, np.ndarray] | None = None


class SelfPlayGame(abc.ABC):
    """One game in progress, as seen by the collector.

    Token blocks are numpy int arrays of shape [tokens, width]. After every
    step the next actor's stream must end exactly at the token whose hidden
    state its policy reads (it must receive at least one new token).
    """

    num_players: int
    rng: random.Random

    @abc.abstractmethod
    def prefix_tokens(self) -> list[np.ndarray]:
        """Each observer's stream up to the first decision."""

    @abc.abstractmethod
    def decision(self) -> Decision:
        """The pending decision; only called while not :meth:`done`."""

    @abc.abstractmethod
    def step(self, action: int) -> list[np.ndarray]:
        """Apply ``action``; return every observer's newly visible tokens.

        When this ends the game the blocks complete each stream; they are
        recorded but never decoded.
        """

    @abc.abstractmethod
    def done(self) -> bool: ...

    @abc.abstractmethod
    def returns(self) -> np.ndarray:
        """Zero-sum terminal reward per absolute seat, float [num_players]."""

    def lifetime(self) -> int:
        """Expected remaining decisions; orders games so retirement is tail-first."""

        return 0

    def context_starts(self) -> list[int | None] | None:
        """Where a fresh context begins in the blocks the last ``step`` returned.

        Per observer, the index into its block from which the policy forgets
        everything before (a Piquet partie restarts each deal), or None to
        continue. The stream keeps every token; decision positions count from
        its start and the game's training sequences split it the same way.
        """

        return None


@dataclass
class DecisionRecord:
    seat: int
    head: str
    position: int  # readout index in the seat's own stream
    action: int
    probs: np.ndarray  # masked behavior policy, float32 [actions]
    features: dict[str, np.ndarray] | None = None
    version: int = 0  # the collector's policy version when sampled
    # The actor's raw player_values at the readout (acting seat first), when
    # the collector records them ([ppo] actor_baseline = "rollout").
    value: np.ndarray | None = None


@dataclass
class Trajectory:
    game: SelfPlayGame
    streams: list[np.ndarray]  # per absolute seat, [length, width]
    decisions: list[DecisionRecord]  # in game order
    returns: np.ndarray


@dataclass
class CollectionStats:
    games: int = 0
    decisions: int = 0
    waves: int = 0
    forward_rows: int = 0
    padded_rows: int = 0
    peak_rows: int = 0
    compactions: int = 0
    collect_sec: float = 0.0
    forward_sec: float = 0.0  # host time preparing and launching device work
    wait_sec: float = 0.0  # host time blocked on the device
    sample_sec: float = 0.0
    step_sec: float = 0.0
    in_flight: int = 0  # games still running when the collection returned
    policy_lag: float = 0.0  # mean versions between sampling and the update


@dataclass
class _Live:
    game: SelfPlayGame
    base: int  # first cache row
    lengths: np.ndarray  # per observer cached tokens
    blocks: list[list[np.ndarray]]  # per observer recorded blocks
    decisions: list[DecisionRecord] = field(default_factory=list)
    pending: Decision | None = None
    readout: np.ndarray | None = None  # the pending actor's last token
    logits: np.ndarray | None = None
    value: np.ndarray | None = None
    # Per observer, stream tokens before its current context (see context_starts).
    offsets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.offsets is None:
            self.offsets = np.zeros_like(self.lengths)


def restart_contexts(state: _Live, blocks: list[np.ndarray]) -> list[np.ndarray]:
    """Apply the game's context restarts after a step; returns the blocks to decode.

    A restarted observer's cached length drops to zero: the new context is
    written from position 0, and causal attention never reaches the stale
    keys past it.
    """

    starts = state.game.context_starts()
    if starts is None:
        return blocks
    decoded = list(blocks)
    for seat, start in enumerate(starts):
        if start is None:
            continue
        state.offsets[seat] += state.lengths[seat] + start
        state.lengths[seat] = 0
        decoded[seat] = blocks[seat][start:]
    return decoded


@dataclass
class _Cohort:
    cache: object
    live: list[tuple[int, _Live]]
    # Launched policy reads: the states and their logits on the way to the host.
    launched: list[tuple[list[_Live], torch.Tensor]] = field(default_factory=list)
    fence: Fence | None = None
    # Host arrays that in-flight non-blocking uploads still read from.
    keep: list[np.ndarray] = field(default_factory=list)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


class SelfPlayCollector:
    ROW_BUCKET = 64
    SPAN_BUCKET = 32

    def __init__(
        self,
        model,
        *,
        device: torch.device,
        precision: str,
        kv_dtype: str,
        max_rows: int,
        max_length: int,
        cohorts: int = 1,
        record_values: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.precision = precision
        self.record_values = record_values
        self.kv_dtype = DTYPES[kv_dtype]
        self.max_rows = max_rows
        self.cohorts = max(1, cohorts)
        # One spare slot past the longest stream absorbs ragged padding.
        self.cache_length = max_length + 1
        self._caches: dict[int, object] = {}
        self.stats = CollectionStats()

    def release(self) -> None:
        """Drop the KV pools; they are dead weight during the update's backward."""

        self._caches = {}
        release_device_memory(self.device)

    @torch.inference_mode()
    def collect(self, games: list[SelfPlayGame]) -> list[Trajectory]:
        started = time.perf_counter()
        self.stats = CollectionStats(games=len(games))
        order = sorted(range(len(games)), key=lambda index: -games[index].lifetime())
        results: dict[int, Trajectory] = {}
        chunk: list[int] = []
        rows = 0
        for index in order:
            players = games[index].num_players
            if chunk and rows + players > self.max_rows:
                self._run(chunk, games, results)
                chunk, rows = [], 0
            chunk.append(index)
            rows += players
        if chunk:
            self._run(chunk, games, results)
        self.stats.collect_sec = time.perf_counter() - started
        return [results[index] for index in range(len(games))]

    # ------------------------------------------------------------------

    def _ensure_cache(self, slot: int, rows: int):
        cache = self._caches.get(slot)
        if cache is None or cache.capacity < rows:
            self._caches.pop(slot, None)
            release_device_memory(self.device)
            capacity = max(rows, min(self.max_rows, rows + rows // 4))
            cache = self.model.new_cache(capacity, self.cache_length, self.kv_dtype)
            self._caches[slot] = cache
        return cache

    def _run(self, indices: list[int], games: list[SelfPlayGame], results: dict[int, Trajectory]) -> None:
        count = min(self.cohorts, len(indices))
        cohorts: list[_Cohort] = []
        total_rows = 0
        for slot in range(count):
            # Interleaving keeps each cohort's lifetimes in descending order.
            members = indices[slot::count]
            rows = sum(games[index].num_players for index in members)
            total_rows += rows
            live: list[tuple[int, _Live]] = []
            row = 0
            for index in members:
                game = games[index]
                live.append(
                    (
                        index,
                        _Live(
                            game=game,
                            base=row,
                            lengths=np.zeros(game.num_players, dtype=np.int64),
                            blocks=[[] for _ in range(game.num_players)],
                        ),
                    )
                )
                row += game.num_players
            cohorts.append(_Cohort(cache=self._ensure_cache(slot, rows), live=live))
        self.stats.peak_rows = max(self.stats.peak_rows, total_rows)
        for cohort in cohorts:
            self._prefill(cohort)
        while any(cohort.live for cohort in cohorts):
            for cohort in cohorts:
                if cohort.live:
                    self._wave(cohort, results)

    def _wave(self, cohort: _Cohort, results: dict[int, Trajectory]) -> None:
        """Receive one cohort's policies, act in every live game, launch the next decode."""

        self.stats.waves += 1
        self._receive(cohort)
        t0 = time.perf_counter()
        self._sample([state for _, state in cohort.live])
        t1 = time.perf_counter()
        appends: list[tuple[_Live, list[np.ndarray]]] = []
        survivors: list[tuple[int, _Live]] = []
        for index, state in cohort.live:
            record = state.decisions[-1]
            blocks = state.game.step(record.action)
            for seat, block in enumerate(blocks):
                if len(block):
                    state.blocks[seat].append(block)
            if state.game.done():
                results[index] = self._finish(state)
            else:
                appends.append((state, restart_contexts(state, blocks)))
                survivors.append((index, state))
        t2 = time.perf_counter()
        self.stats.sample_sec += t1 - t0
        self.stats.step_sec += t2 - t1
        if len(survivors) != len(cohort.live):
            self._compact(cohort, [state for _, state in survivors])
        cohort.live = survivors
        if appends:
            self._append(cohort, appends)

    def _finish(self, state: _Live) -> Trajectory:
        streams = [np.concatenate(blocks, axis=0) for blocks in state.blocks]
        return Trajectory(
            game=state.game,
            streams=streams,
            decisions=state.decisions,
            returns=np.asarray(state.game.returns(), dtype=np.float64),
        )

    def _upload(self, cohort: _Cohort, array: np.ndarray) -> torch.Tensor:
        cohort.keep.append(array)
        return to_device(array, self.device)

    def _compact(self, cohort: _Cohort, survivors: list[_Live]) -> None:
        """Slide survivors down so live rows stay a dense prefix of the pool."""

        sources: list[int] = []
        targets: list[int] = []
        length = 0
        row = 0
        for state in survivors:
            if state.base != row:
                for seat in range(state.game.num_players):
                    sources.append(state.base + seat)
                    targets.append(row + seat)
                length = max(length, int(state.lengths.max()))
                state.base = row
            row += state.game.num_players
        if sources:
            self.stats.compactions += 1
            cohort.cache.copy_rows(
                self._upload(cohort, np.asarray(sources, dtype=np.int64)),
                self._upload(cohort, np.asarray(targets, dtype=np.int64)),
                length,
            )

    def _prefill(self, cohort: _Cohort) -> None:
        """Encode every prefix; rows sharing a prefix length share one forward."""

        started = time.perf_counter()
        groups: dict[int, list[tuple[_Live, int, np.ndarray]]] = {}
        for _, state in cohort.live:
            state.pending = state.game.decision()
            for seat, prefix in enumerate(state.game.prefix_tokens()):
                groups.setdefault(prefix.shape[0], []).append((state, seat, prefix))
                state.blocks[seat].append(prefix)
                state.lengths[seat] = prefix.shape[0]
        for length, entries in groups.items():
            tokens = self._upload(cohort, np.stack([prefix for _, _, prefix in entries]).astype(np.int64))
            rows = self._upload(
                cohort, np.asarray([state.base + seat for state, seat, _ in entries], dtype=np.int64)
            )
            readout = [
                (position, state)
                for position, (state, seat, _) in enumerate(entries)
                if state.pending.seat == seat
            ]
            for position, state in readout:
                state.readout = entries[position][2][-1]
            with autocast_context(self.device, self.precision):
                hidden = self.model.forward_prefill(tokens, cohort.cache, rows)
            self.stats.forward_rows += len(entries) * length
            if readout:
                index = self._upload(cohort, np.asarray([p for p, _ in readout], dtype=np.int64))
                self._launch_policies(cohort, hidden[index], [state for _, state in readout])
        cohort.fence = Fence(self.device)
        self.stats.forward_sec += time.perf_counter() - started

    def _append(self, cohort: _Cohort, appends: list[tuple[_Live, list[np.ndarray]]]) -> None:
        started = time.perf_counter()
        run = max(len(block) for _, blocks in appends for block in blocks)
        live_rows = appends[-1][0].base + appends[-1][0].game.num_players
        rows = max(live_rows, min(_round_up(live_rows, self.ROW_BUCKET), cohort.cache.capacity))
        width = next(block.shape[1] for _, blocks in appends for block in blocks if len(block))
        # Rows past the live prefix are shape padding: zeros, written at position 0.
        tokens = np.zeros((rows, run, width), dtype=np.int64)
        positions = np.zeros(rows, dtype=np.int64)
        offsets = np.zeros(rows, dtype=np.int64)
        readout_rows: list[int] = []
        readout_states: list[_Live] = []
        longest = 0
        for state, blocks in appends:
            for seat, block in enumerate(blocks):
                row = state.base + seat
                count = len(block)
                positions[row] = state.lengths[seat]
                if count:
                    tokens[row, :count] = block
                    # Repeat the last token as padding; its output is dropped.
                    tokens[row, count:] = block[-1]
                    offsets[row] = count - 1
                state.lengths[seat] += count
            longest = max(longest, int(state.lengths.max()))
            state.pending = state.game.decision()
            if not len(blocks[state.pending.seat]):
                raise AssertionError("The next actor's stream received no readout token.")
            state.readout = blocks[state.pending.seat][-1]
            readout_rows.append(state.base + state.pending.seat)
            readout_states.append(state)
        if longest >= self.cache_length:
            raise RuntimeError("A stream outgrew the model's maximum sequence length.")
        with autocast_context(self.device, self.precision):
            hidden = self.model.forward_ragged(
                self._upload(cohort, tokens),
                self._upload(cohort, positions),
                cohort.cache,
                self._upload(cohort, offsets),
                span=_round_up(int(positions.max()) + run, self.SPAN_BUCKET),
            )
        self.stats.forward_rows += rows * run
        self.stats.padded_rows += rows * run - sum(
            len(block) for _, blocks in appends for block in blocks
        )
        index = self._upload(cohort, np.asarray(readout_rows, dtype=np.int64))
        self._launch_policies(cohort, hidden[index], readout_states)
        cohort.fence = Fence(self.device)
        self.stats.forward_sec += time.perf_counter() - started

    def _launch_policies(self, cohort: _Cohort, hidden: torch.Tensor, states: list[_Live]) -> None:
        """Queue each pending decision's head on its actor's hidden state."""

        cohort.launched.extend(
            launch_policies(
                self.model, self.device, self.precision, lambda array: self._upload(cohort, array),
                hidden, states, None, values=self.record_values,
            )
        )

    def _receive(self, cohort: _Cohort) -> None:
        """Wait for this cohort's launched policies and hand each state its logits."""

        started = time.perf_counter()
        if cohort.fence is not None:
            cohort.fence.wait()
        self.stats.wait_sec += time.perf_counter() - started
        receive_launched(cohort.launched)
        cohort.launched = []
        cohort.keep = []
        cohort.fence = None

    def _sample(self, states: list[_Live]) -> None:
        self.stats.decisions += sample_decisions(states)


DECISION_BUCKET = 16
CANDIDATE_BUCKET = 8


def launch_policies(model, device, precision, upload, hidden, states, context, *, values: bool = False):
    """Queue each pending decision's head; returns [(states, logits, values or None)] on their way to the host.

    ``hidden`` [N, d] and ``context`` (None or [N, ...]) are the states'
    readouts in order. Reads are padded to bucketed decision and candidate
    counts (MPS compiles a graph per shape) by repeating the first decision.
    With ``values`` the actor's ``player_values`` at the same readouts come
    along (the rollout-recorded baseline).
    """

    by_head: dict[str, list[int]] = {}
    for position, state in enumerate(states):
        by_head.setdefault(state.pending.head, []).append(position)
    launched = []
    with autocast_context(device, precision):
        for head, positions in by_head.items():
            padded = positions + [positions[0]] * (_round_up(len(positions), DECISION_BUCKET) - len(positions))
            features = _stack_features([states[p].pending.features for p in padded], CANDIDATE_BUCKET) or {}
            # Readout tokens only matter to a head without the embedding's context.
            extra = [] if context is not None else [np.stack([states[p].readout for p in padded]).astype(np.int64)]
            index, *uploaded = upload_packed(upload, [np.asarray(padded, dtype=np.int64), *features.values(), *extra])
            tokens = uploaded.pop() if extra else None
            features = dict(zip(features, uploaded)) or None
            logits = model.policy_logits(
                hidden[index], head, features, tokens, None if context is None else context[index]
            )
            seat_values = (
                model.player_values(hidden[index]).float().to("cpu", non_blocking=True) if values else None
            )
            launched.append(
                ([states[p] for p in positions], logits.float().to("cpu", non_blocking=True), seat_values)
            )
    return launched


def receive_launched(launched) -> None:
    """Hand each state its logits (and recorded values) once the launch's fence has passed."""

    for states, logits, seat_values in launched:
        rows = logits.numpy()
        recorded = None if seat_values is None else seat_values.numpy()
        for row, state in enumerate(states):
            state.logits = rows[row]
            state.value = None if recorded is None else recorded[row].copy()


def sample_decisions(states, version: int = 0) -> int:
    """Sample every state's pending decision from its logits; record it. Returns the count."""

    by_head: dict[str, list] = {}
    for state in states:
        by_head.setdefault(state.pending.head, []).append(state)
    for head, group in by_head.items():
        width = max(state.pending.legal.shape[0] for state in group)
        logits = np.full((len(group), width), -np.inf, dtype=np.float32)
        mask = np.zeros((len(group), width), dtype=bool)
        for row, state in enumerate(group):
            legal = state.pending.legal
            logits[row, : legal.shape[0]] = state.logits[: legal.shape[0]]
            mask[row, : legal.shape[0]] = legal
        probs = masked_probabilities(logits, mask)
        uniforms = np.fromiter((state.game.rng.random() for state in group), dtype=np.float64)
        sampled = sample_rows(probs, mask, uniforms)
        for row, state in enumerate(group):
            decision = state.pending
            actions = decision.legal.shape[0]
            state.decisions.append(
                DecisionRecord(
                    seat=decision.seat,
                    head=head,
                    position=int(state.offsets[decision.seat] + state.lengths[decision.seat]) - 1,
                    action=int(sampled[row]),
                    probs=probs[row, :actions].astype(np.float32),
                    features=decision.features,
                    version=version,
                    value=state.value,
                )
            )
            state.pending = None
            state.logits = None
            state.value = None
    return len(states)


def _stack_features(
    features: list[dict[str, np.ndarray] | None], bucket: int = 1
) -> dict[str, np.ndarray] | None:
    """Pad per-decision candidate features along the candidate axis and stack."""

    if features[0] is None:
        return None
    count = _round_up(max(next(iter(entry.values())).shape[0] for entry in features), bucket)
    stacked: dict[str, np.ndarray] = {}
    for key in features[0]:
        first = features[0][key]
        array = np.zeros((len(features), count, *first.shape[1:]), dtype=first.dtype)
        for row, entry in enumerate(features):
            value = entry[key]
            array[row, : value.shape[0]] = value
        stacked[key] = array
    return stacked
