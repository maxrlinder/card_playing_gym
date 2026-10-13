"""Cached decoding must reproduce the full causal forward exactly (fp32, CPU)."""

from __future__ import annotations

import random

import numpy as np
import torch

from cardgym.games.plump.env import PlumpEnv
from cardgym.games.plump.model import PlumpActor
from cardgym.games.plump.rounds import RoundSpec, round_game_config
from cardgym.games.plump.tokens import TOKEN_WIDTH, Schema, seat_tokens
from cardgym.nn.kv_cache import KVCache
from tests.conftest import randomize, tiny_plump_model


def streams(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = random.Random(seed)
    result = []
    for index in range(count):
        players, hand = rng.choice([3, 4, 5]), rng.choice(range(3, 11))
        start = rng.randrange(players)
        env = PlumpEnv(round_game_config(RoundSpec(players, hand), bidding_start_player=start), seed=index)
        env.reset(seed=index)
        while not env.is_done():
            env.step(rng.choice(env.legal_actions()))
        observer = rng.randrange(players)
        result.append(
            seat_tokens(
                Schema(), env.state.event_log, observer, players, hand,
                env.state.current_round.initial_hands[observer], start,
            )
        )
    return result


def test_prefill_and_ragged_decode_match_the_full_forward():
    model = randomize(PlumpActor(tiny_plump_model(value_embeddings="all"))).eval()
    batch = streams(6)
    lengths = [stream.shape[0] for stream in batch]
    padded = np.zeros((len(batch), max(lengths), TOKEN_WIDTH), dtype=np.int64)
    for row, stream in enumerate(batch):
        padded[row, : stream.shape[0]] = stream
    with torch.no_grad():
        full = model.forward_hidden(torch.from_numpy(padded))
        cache = model.new_cache(len(batch), max(lengths) + 1)
        prefix = 4
        hidden = model.forward_prefill(torch.from_numpy(padded[:, :prefix]), cache, None)
        torch.testing.assert_close(hidden, full[:, prefix - 1], rtol=1e-5, atol=1e-5)
        rng = random.Random(1)
        cursor = [prefix] * len(batch)
        while any(c < n for c, n in zip(cursor, lengths)):
            counts = [min(rng.randint(0, 3), n - c) for c, n in zip(cursor, lengths)]
            if not any(counts):
                continue
            run = max(counts)
            tokens = np.zeros((len(batch), run, TOKEN_WIDTH), dtype=np.int64)
            offsets = np.zeros(len(batch), dtype=np.int64)
            for row, (start, count) in enumerate(zip(cursor, counts)):
                if not count:
                    continue  # zero padding, as the collector does
                block = padded[row, start : start + count]
                tokens[row, :count] = block
                tokens[row, count:] = block[-1]
                offsets[row] = count - 1
            hidden = model.forward_ragged(
                torch.from_numpy(tokens),
                torch.tensor(cursor),
                cache,
                torch.from_numpy(offsets),
            )
            for row, (start, count) in enumerate(zip(cursor, counts)):
                if count:
                    torch.testing.assert_close(
                        hidden[row], full[row, start + count - 1], rtol=1e-5, atol=1e-5
                    )
            cursor = [c + n for c, n in zip(cursor, counts)]


def test_kv_cache_copy_rows_moves_prefixes():
    cache = KVCache(n_layers=2, kv_heads=1, head_dim=2, capacity=4, max_len=5, device="cpu")
    for layer in range(2):
        cache.k[layer][:] = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
        cache.v[layer][:] = -torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
    cache.copy_rows(torch.tensor([2, 3]), torch.tensor([1, 2]), length=3)
    assert cache.k[1][:, 0, 0, 0].tolist() == [0.0, 2.0, 3.0, 3.0]
    assert cache.v[0][1, 0, 3, 0].item() == -1.0  # beyond the copied length


def test_conditioning_starts_at_zero():
    """PLE and value embeddings are additive capacity that starts switched off."""

    config = tiny_plump_model()
    with_conditioning = PlumpActor(config).eval()
    plain = PlumpActor(tiny_plump_model(per_layer_dim=0, value_embeddings="off")).eval()
    plain.load_state_dict(
        {k: v for k, v in with_conditioning.state_dict().items() if k in plain.state_dict()}
    )
    tokens = torch.from_numpy(np.stack(streams(1)))
    with torch.no_grad():
        torch.testing.assert_close(
            with_conditioning.forward_hidden(tokens), plain.forward_hidden(tokens)
        )
