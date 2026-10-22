"""A continuous collector that decodes each seat with its own model.

Every game names the model of each seat in ``game.seat_models`` (indexes
into ``models``). All models share one KV cache, so they must share the
cache layout (same trunk shape). Used by head-to-head evaluations
(tools/tournament.py, ``cardgym piquet evaluate --against``).
"""

from __future__ import annotations

import numpy as np

from cardgym.core.devices import Fence, autocast_context, upload_packed

from .continuous import ContinuousCollector
from .selfplay import _round_up, launch_policies


class MatchCollector(ContinuousCollector):
    """The continuous collector with a model per seat (``game.seat_models`` indexes ``models``)."""

    def __init__(self, models, temperatures=None, **kwargs) -> None:
        super().__init__(models[0], **kwargs)
        self.models = models
        self.temperatures = temperatures or [1.0] * len(models)

    def _receive(self, cohort: int) -> None:
        launched = self._waves[cohort].launched
        super()._receive(cohort)
        for states, *_ in launched:
            for state in states:
                temperature = self.temperatures[state.game.seat_models[state.pending.seat]]
                if temperature == 0.0:  # argmax: all probability on the best legal action
                    legal = state.pending.legal
                    best = int(np.argmax(np.where(legal, state.logits[: len(legal)], -np.inf)))
                    state.logits = np.full_like(state.logits, -np.inf)
                    state.logits[best] = 0.0
                elif temperature != 1.0:
                    state.logits = state.logits / temperature

    def _launch(self, cohort: int) -> bool:
        groups: dict[int, tuple[list, list]] = {}
        for slot in self._slots:
            state = slot.state
            if state is None or slot.cohort != cohort:
                continue
            actor = state.pending.seat
            for seat, block in slot.queued:
                entries, readouts = groups.setdefault(state.game.seat_models[seat], ([], []))
                if seat == actor:
                    readouts.append((len(entries), state))
                entries.append((slot.base + seat, int(state.lengths[seat]), block))
                state.lengths[seat] += len(block)
            if int(state.lengths.max()) >= self.cache_length:
                raise RuntimeError("A stream outgrew the model's maximum sequence length.")
            slot.queued = []
        if not groups:
            return False
        upload = lambda array: self._upload(array, cohort)  # noqa: E731
        wave = self._waves[cohort]
        for model_index, (entries, readouts) in sorted(groups.items()):
            model = self.models[model_index]
            run = max(len(block) for _, _, block in entries)
            rows = _round_up(len(entries), self.ROW_BUCKET)
            width = entries[0][2].shape[1]
            tokens = np.zeros((rows, run, width), dtype=np.int64)
            cache_rows = np.full(rows, self._scratch, dtype=np.int64)
            positions = np.zeros(rows, dtype=np.int64)
            offsets = np.zeros(rows, dtype=np.int64)
            for index, (row, position, block) in enumerate(entries):
                tokens[index, : len(block)] = block
                tokens[index, len(block) :] = block[-1]
                cache_rows[index] = row
                positions[index] = position
                offsets[index] = len(block) - 1
            span = _round_up(int(positions.max()) + run, self.SPAN_BUCKET)
            read = np.asarray([entry for entry, _ in readouts], dtype=np.int64)
            tokens_d, positions_d, offsets_d, rows_d, index, read_offsets = upload_packed(
                upload, [tokens, positions, offsets, cache_rows, read, offsets[read]]
            )
            with autocast_context(self.device, self.precision):
                hidden = model.forward_ragged(tokens_d, positions_d, self._cache, offsets_d, span=span, rows=rows_d)
            context = model.pop_context()
            if not readouts:
                continue
            if context is not None:
                context = context[index, read_offsets]
            wave.launched += launch_policies(
                model, self.device, self.precision, upload, hidden[index], [state for _, state in readouts], context
            )
        wave.fence = Fence(self.device)
        self.stats.waves += 1
        return True
