"""Round-robin between Rummy checkpoints on duplicate deals, until the error bars are small.

Every pair of checkpoints (A, B) plays the same seeded deals in every seating:
one A seat against P - 1 B seats and one B seat against P - 1 A seats, the
lone seat rotated through every position (heads-up: both seat orders). The
unit of evidence is a deal: A's mean score minus B's mean score, averaged over
its seatings, so the luck of the cards cancels and only the play differs.
Deals cover the training mix (2-4 players x 1-2 decks, seven cards).

Rounds of fresh deals run until every pair of neighbouring checkpoints is
either separated by ``--sigmas`` standard errors or measured to within
``--resolution`` points, or ``--max-rounds`` is reached. After each round the
pair matrix and least-squares ratings (points per game relative to the first
checkpoint) are printed and written to ``--output``.

All models share one KV cache (same architecture): each seat's rows are
decoded by the model playing that seat. Policies sample, as in training
(``--temperatures`` sharpens or flattens a checkpoint's sampling; 0 plays
the argmax).

    .venv/bin/python tools/tournament.py runs/rummy/r7/checkpoints/iter_{000000,002700,005300,008000,010600}.pt
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardgym.core.devices import resolve_device  # noqa: E402
from cardgym.games.rummy.evaluation import load_actor  # noqa: E402
from cardgym.games.rummy.selfplay import RummySelfPlay  # noqa: E402
from cardgym.games.rummy.tokens import max_stream_length  # noqa: E402
from cardgym.nn.metal_trunk import unsupported_reason  # noqa: E402
from cardgym.rl.continuous import Lane  # noqa: E402
from cardgym.rl.match import MatchCollector  # noqa: E402

MAX_TURNS = 200
HAND_SIZE = 7


def seatings(players: int) -> list[tuple[int, bool]]:
    """(lone seat, lone seat is A): heads-up needs only A's seat to rotate."""

    sides = (True,) if players == 2 else (True, False)
    return [(lone, a_alone) for lone in range(players) for a_alone in sides]


def play_round(collector, pairs, configs, deals: int, rng: random.Random, in_flight: int):
    """Returns {(pair, config): [per-deal A-minus-B points]} for ``deals`` fresh deals each."""

    lanes, units = [], {}
    for players, decks in configs:
        games: deque = deque()
        for pair_index, (a, b) in enumerate(pairs):
            for deal in range(deals):
                seed = rng.getrandbits(48)
                key = (pair_index, players, decks, deal)
                units[key] = []
                for lone, a_alone in seatings(players):
                    game = RummySelfPlay(players, HAND_SIZE, seed=seed, decks=decks, max_turns=MAX_TURNS)
                    lone_model, crowd_model = (a, b) if a_alone else (b, a)
                    game.seat_models = tuple(lone_model if seat == lone else crowd_model for seat in range(players))
                    game.unit = key
                    games.append(game)
        lanes.append(Lane(f"{players}p{decks}d", players, in_flight, None, lambda _rng, g=games: g.popleft() if g else None))
    trajectories = collector.collect(lanes, rng)
    collector.reset()
    truncated = 0
    for trajectory in trajectories:
        game = trajectory.game
        a, b = pairs[game.unit[0]]
        scores = game.engine.scores()
        seats_a = [s for s, m in enumerate(game.seat_models) if m == a]
        seats_b = [s for s, m in enumerate(game.seat_models) if m == b]
        units[game.unit].append(float(scores[seats_a].mean() - scores[seats_b].mean()))
        truncated += game.engine.truncated
    results: dict[tuple[int, int, int], list[float]] = {}
    for (pair_index, players, decks, _), diffs in units.items():
        assert len(diffs) == len(seatings(players))
        results.setdefault((pair_index, players, decks), []).append(float(np.mean(diffs)))
    return results, len(trajectories), truncated


def summarise(samples, pairs, configs, labels):
    """Pair means with standard errors (configs weighted equally) and least-squares ratings."""

    pair_stats = []
    for pair_index in range(len(pairs)):
        means, variances = [], []
        for players, decks in configs:
            values = np.asarray(samples[(pair_index, players, decks)])
            means.append(values.mean())
            variances.append(values.var(ddof=1) / len(values))
        pair_stats.append((float(np.mean(means)), float(math.sqrt(sum(variances)) / len(configs))))
    count = len(labels)
    # Weighted least squares on r_a - r_b = mean(a, b), with r_0 = 0.
    design = np.zeros((len(pairs), count - 1))
    target = np.zeros(len(pairs))
    weight = np.zeros(len(pairs))
    for row, ((a, b), (mean, se)) in enumerate(zip(pairs, pair_stats)):
        if a:
            design[row, a - 1] += 1
        if b:
            design[row, b - 1] -= 1
        target[row], weight[row] = mean, 1 / max(se, 1e-9) ** 2
    normal = design.T @ (design * weight[:, None])
    covariance = np.linalg.inv(normal)
    ratings = np.concatenate([[0.0], covariance @ design.T @ (weight * target)])
    rating_se = np.concatenate([[0.0], np.sqrt(np.diag(covariance))])
    return pair_stats, ratings, rating_se


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--deals", type=int, default=16, help="fresh deals per pair and config per round")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--sigmas", type=float, default=3.0)
    parser.add_argument("--resolution", type=float, default=0.5, help="points: a neighbour gap known this well is done")
    parser.add_argument("--in-flight", type=int, default=96, help="games in flight per (players, decks) lane")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--temperatures", default="", help="comma-separated sampling temperature per checkpoint (default 1)"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device("auto")
    models, labels = [], []
    for path in args.checkpoints:
        actor, iteration = load_actor(path, device)
        if device.type == "mps" and unsupported_reason(actor.trunk_config) is None:
            actor.enable_metal_trunk(max_stream_length(MAX_TURNS))
        models.append(actor)
        labels.append(str(iteration))
    temperatures = [float(t) for t in args.temperatures.split(",")] if args.temperatures else [1.0] * len(models)
    if len(temperatures) != len(models):
        raise ValueError("--temperatures needs one value per checkpoint.")
    labels = [label if t == 1.0 else f"{label}@T{t:g}" for label, t in zip(labels, temperatures)]
    pairs = list(itertools.combinations(range(len(models)), 2))
    configs = [(players, decks) for players in (2, 3, 4) for decks in (1, 2)]
    collector = MatchCollector(
        models, temperatures, device=device, precision="bf16", kv_dtype="fp16", max_length=max_stream_length(MAX_TURNS), cohorts=2
    )
    rng = random.Random(args.seed)
    samples: dict = {}
    neighbours = [pairs.index((i, i + 1)) for i in range(len(models) - 1)]
    started = time.perf_counter()
    games = 0
    for round_index in range(1, args.max_rounds + 1):
        results, played, truncated = play_round(collector, pairs, configs, args.deals, rng, args.in_flight)
        games += played
        for key, values in results.items():
            samples.setdefault(key, []).extend(values)
        pair_stats, ratings, rating_se = summarise(samples, pairs, configs, labels)
        deals = len(samples[(0, *configs[0])])
        print(
            f"\n== round {round_index}: {deals} deals per pair and config, {games} games,"
            f" {time.perf_counter() - started:.0f}s (truncated this round: {truncated}/{played})"
        )
        print("rating (points per game vs iteration " + labels[0] + "):")
        for label, rating, se in zip(labels, ratings, rating_se):
            print(f"  iter {label:>6}: {rating:+7.2f} +- {se:.2f}")
        print("pairs (A - B points per game, duplicate deals):")
        for (a, b), (mean, se) in zip(pairs, pair_stats):
            print(f"  {labels[a]:>6} vs {labels[b]:>6}: {mean:+7.2f} +- {se:.2f}  ({abs(mean) / max(se, 1e-9):.1f} sigma)")
        if args.output:
            args.output.write_text(json.dumps({
                "labels": labels, "rounds": round_index, "deals": deals, "games": games,
                "ratings": ratings.tolist(), "rating_se": rating_se.tolist(),
                "pairs": [{"a": labels[a], "b": labels[b], "mean": m, "se": s} for (a, b), (m, s) in zip(pairs, pair_stats)],
            }, indent=2))
        done = all(
            abs(pair_stats[p][0]) >= args.sigmas * pair_stats[p][1] or pair_stats[p][1] <= args.resolution
            for p in neighbours
        )
        if done:
            print("\nEvery neighbouring pair is resolved.")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
