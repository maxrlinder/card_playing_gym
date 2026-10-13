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

from cardgym.core.devices import DTYPES, autocast_context, release_device_memory

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


@dataclass
class DecisionRecord:
    seat: int
    head: str
    position: int  # readout index in the seat's own stream
    action: int
    probs: np.ndarray  # masked behavior policy, float32 [actions]
    features: dict[str, np.ndarray] | None = None


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
    forward_sec: float = 0.0
    sample_sec: float = 0.0
    step_sec: float = 0.0


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


class SelfPlayCollector:
    def __init__(
        self,
        model,
        *,
        device: torch.device,
        precision: str,
        kv_dtype: str,
        max_rows: int,
        max_length: int,
    ) -> None:
        self.model = model
        self.device = device
        self.precision = precision
        self.kv_dtype = DTYPES[kv_dtype]
        self.max_rows = max_rows
        # One spare slot past the longest stream absorbs ragged padding.
        self.cache_length = max_length + 1
        self._cache = None
        self.stats = CollectionStats()

    def release(self) -> None:
        """Drop the KV pool; it is dead weight during the update's backward."""

        self._cache = None
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

    def _ensure_cache(self, rows: int):
        cache = self._cache
        if cache is None or cache.capacity < rows:
            self._cache = None
            release_device_memory(self.device)
            capacity = max(rows, min(self.max_rows, rows + rows // 4))
            self._cache = self.model.new_cache(capacity, self.cache_length, self.kv_dtype)
        return self._cache

    def _run(self, indices: list[int], games: list[SelfPlayGame], results: dict[int, Trajectory]) -> None:
        total_rows = sum(games[index].num_players for index in indices)
        cache = self._ensure_cache(total_rows)
        live: list[tuple[int, _Live]] = []
        row = 0
        for index in indices:
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
        self.stats.peak_rows = max(self.stats.peak_rows, total_rows)
        self._prefill([state for _, state in live], cache)
        while live:
            self.stats.waves += 1
            t0 = time.perf_counter()
            self._sample([state for _, state in live])
            t1 = time.perf_counter()
            appends: list[tuple[_Live, list[np.ndarray]]] = []
            survivors: list[tuple[int, _Live]] = []
            for index, state in live:
                record = state.decisions[-1]
                blocks = state.game.step(record.action)
                for seat, block in enumerate(blocks):
                    if len(block):
                        state.blocks[seat].append(block)
                if state.game.done():
                    results[index] = self._finish(state)
                else:
                    appends.append((state, blocks))
                    survivors.append((index, state))
            t2 = time.perf_counter()
            self.stats.sample_sec += t1 - t0
            self.stats.step_sec += t2 - t1
            if len(survivors) != len(live):
                self._compact(cache, [state for _, state in survivors])
            live = survivors
            if appends:
                self._append(appends, cache)

    def _finish(self, state: _Live) -> Trajectory:
        streams = [np.concatenate(blocks, axis=0) for blocks in state.blocks]
        return Trajectory(
            game=state.game,
            streams=streams,
            decisions=state.decisions,
            returns=np.asarray(state.game.returns(), dtype=np.float64),
        )

    def _compact(self, cache, survivors: list[_Live]) -> None:
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
            cache.copy_rows(
                torch.tensor(sources, dtype=torch.long, device=self.device),
                torch.tensor(targets, dtype=torch.long, device=self.device),
                length,
            )

    def _prefill(self, states: list[_Live], cache) -> None:
        """Encode every prefix; rows sharing a prefix length share one forward."""

        groups: dict[int, list[tuple[_Live, int, np.ndarray]]] = {}
        for state in states:
            state.pending = state.game.decision()
            for seat, prefix in enumerate(state.game.prefix_tokens()):
                groups.setdefault(prefix.shape[0], []).append((state, seat, prefix))
                state.blocks[seat].append(prefix)
                state.lengths[seat] = prefix.shape[0]
        for length, entries in groups.items():
            tokens = torch.from_numpy(np.stack([prefix for _, _, prefix in entries])).to(self.device)
            rows = torch.tensor(
                [state.base + seat for state, seat, _ in entries], dtype=torch.long, device=self.device
            )
            readout = [
                (position, state)
                for position, (state, seat, _) in enumerate(entries)
                if state.pending.seat == seat
            ]
            for position, state in readout:
                state.readout = entries[position][2][-1]
            with autocast_context(self.device, self.precision):
                hidden = self.model.forward_prefill(tokens, cache, rows)
            self.stats.forward_rows += len(entries) * length
            self._read_policies(
                hidden[torch.tensor([p for p, _ in readout], dtype=torch.long, device=self.device)],
                [state for _, state in readout],
            )

    def _append(self, appends: list[tuple[_Live, list[np.ndarray]]], cache) -> None:
        started = time.perf_counter()
        run = max(len(block) for _, blocks in appends for block in blocks)
        live_rows = appends[-1][0].base + appends[-1][0].game.num_players
        width = next(block.shape[1] for _, blocks in appends for block in blocks if len(block))
        tokens = np.zeros((live_rows, run, width), dtype=np.int64)
        positions = np.zeros(live_rows, dtype=np.int64)
        offsets = np.zeros(live_rows, dtype=np.int64)
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
                torch.from_numpy(tokens).to(self.device),
                torch.from_numpy(positions).to(self.device),
                cache,
                torch.from_numpy(offsets).to(self.device),
            )
        self.stats.forward_rows += live_rows * run
        self.stats.padded_rows += live_rows * run - sum(
            len(block) for _, blocks in appends for block in blocks
        )
        self._read_policies(
            hidden[torch.tensor(readout_rows, dtype=torch.long, device=self.device)],
            readout_states,
        )
        self.stats.forward_sec += time.perf_counter() - started

    def _read_policies(self, hidden: torch.Tensor, states: list[_Live]) -> None:
        """Evaluate each pending decision's head on its actor's hidden state."""

        by_head: dict[str, list[int]] = {}
        for position, state in enumerate(states):
            by_head.setdefault(state.pending.head, []).append(position)
        launched = []
        with autocast_context(self.device, self.precision):
            for head, positions in by_head.items():
                features = _stack_features(
                    [states[p].pending.features for p in positions], self.device
                )
                index = torch.tensor(positions, dtype=torch.long, device=self.device)
                tokens = torch.from_numpy(
                    np.stack([states[p].readout for p in positions]).astype(np.int64)
                ).to(self.device)
                launched.append(
                    (positions, self.model.policy_logits(hidden[index], head, features, tokens))
                )
        for positions, logits in launched:
            values = logits.float().cpu().numpy()
            for row, position in enumerate(positions):
                states[position].logits = values[row]

    def _sample(self, states: list[_Live]) -> None:
        by_head: dict[str, list[_Live]] = {}
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
                        position=int(state.lengths[decision.seat]) - 1,
                        action=int(sampled[row]),
                        probs=probs[row, :actions].astype(np.float32),
                        features=decision.features,
                    )
                )
                state.pending = None
                state.logits = None
            self.stats.decisions += len(group)


def _stack_features(
    features: list[dict[str, np.ndarray] | None], device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Pad per-decision candidate features along the candidate axis and stack."""

    if features[0] is None:
        return None
    count = max(next(iter(entry.values())).shape[0] for entry in features)
    stacked: dict[str, torch.Tensor] = {}
    for key in features[0]:
        first = features[0][key]
        array = np.zeros((len(features), count, *first.shape[1:]), dtype=first.dtype)
        for row, entry in enumerate(features):
            value = entry[key]
            array[row, : value.shape[0]] = value
        stacked[key] = torch.from_numpy(array).to(device)
    return stacked
