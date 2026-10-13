"""Paired evaluation of a model against a rule-based opponent.

A seeded deal bank is played with the seats split between the model and the
opponent, in both directions: one model seat against P - 1 opponents, and one
opponent against P - 1 model seats, rotating the lone seat through every
position. A lone seat is structurally different from a crowd, so only the two
directions together say whether the model is better. The headline is the
model seats' mean relative reward (score minus the mean opponent score).

Decoding reuses the self-play collector: rule-based seats act inside a game's
``step``, so their events reach the model's streams as ordinary tokens.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from cardgym.core.io import atomic_write_json, timestamp
from cardgym.rl.checkpoint import load_weights
from cardgym.rl.selfplay import Decision, SelfPlayCollector, SelfPlayGame

from .heuristic import HeuristicPolicy, RandomPolicy
from .model import RummyActor, RummyModelConfig
from .selfplay import RummySelfPlay
from .tokens import max_stream_length


class MixedGame(SelfPlayGame):
    """A round in which only ``model_seats`` ask the collector for decisions."""

    def __init__(self, inner: RummySelfPlay, model_seats: frozenset[int], opponent) -> None:
        self.inner = inner
        self.num_players = inner.num_players
        self.rng = inner.rng
        self.model_seats = model_seats
        self.opponent = opponent
        prefix = inner.prefix_tokens()
        more = self._advance()
        self._prefix = [np.concatenate([prefix[seat], *more[seat]]) for seat in range(self.num_players)]

    def _advance(self) -> list[list[np.ndarray]]:
        blocks: list[list[np.ndarray]] = [[] for _ in range(self.num_players)]
        engine = self.inner.engine
        while not engine.done and engine.current not in self.model_seats:
            for seat, block in enumerate(self.inner.step(self.opponent.act(engine))):
                blocks[seat].append(block)
        return blocks

    def lifetime(self) -> int:
        return self.inner.lifetime()

    def prefix_tokens(self) -> list[np.ndarray]:
        return self._prefix

    def decision(self) -> Decision:
        return self.inner.decision()

    def step(self, action: int) -> list[np.ndarray]:
        first = self.inner.step(action)
        more = self._advance()
        return [np.concatenate([first[seat], *more[seat]]) for seat in range(self.num_players)]

    def done(self) -> bool:
        return self.inner.done()

    def returns(self) -> np.ndarray:
        return self.inner.returns()


def load_actor(path: Path, device: torch.device) -> tuple[RummyActor, int]:
    from cardgym.core.config import parse_section

    weights = load_weights(path, "rummy")
    actor = RummyActor(parse_section(RummyModelConfig, weights["model"], "model"))
    actor.load_state_dict(weights["actor"])
    return actor.to(device).eval(), weights["iteration"]


def evaluate(
    actor: RummyActor,
    *,
    opponent: str,
    player_counts: tuple[int, ...],
    hand_size: int,
    deals: int,
    seed: int,
    device: torch.device,
    precision: str = "fp32",
    opening_points: int = 30,
    max_turns: int = 200,
    max_rows: int = 4096,
) -> dict:
    rng = random.Random(seed)
    games: list[MixedGame] = []
    meta: list[tuple[int, str, frozenset[int]]] = []
    finished: list[tuple[int, str, frozenset[int], np.ndarray, RummySelfPlay]] = []
    for players in player_counts:
        for _ in range(deals):
            deal_seed = rng.getrandbits(48)
            for lone in range(players):
                for direction in _directions(players):
                    seats = frozenset([lone]) if direction == "model_alone" else frozenset(range(players)) - {lone}
                    policy = HeuristicPolicy() if opponent == "heuristic" else RandomPolicy(deal_seed ^ lone)
                    inner = RummySelfPlay(
                        players, hand_size, seed=deal_seed, opening_points=opening_points, max_turns=max_turns
                    )
                    game = MixedGame(inner, seats, policy)
                    if game.done():
                        finished.append((players, direction, seats, game.returns(), inner))
                    else:
                        games.append(game)
                        meta.append((players, direction, seats))
    collector = SelfPlayCollector(
        actor,
        device=device,
        precision=precision,
        kv_dtype=precision,
        max_rows=max_rows,
        max_length=max_stream_length(max_turns),
    )
    trajectories = collector.collect(games)
    collector.release()
    for (players, direction, seats), trajectory, game in zip(meta, trajectories, games):
        finished.append((players, direction, seats, trajectory.returns, game.inner))
    return report(finished, player_counts)


def _directions(players: int) -> tuple[str, ...]:
    # Heads-up, "one model seat" and "one opponent seat" are the same games.
    return ("model_alone",) if players == 2 else ("model_alone", "opponent_alone")


def report(finished, player_counts: tuple[int, ...]) -> dict:
    result: dict = {"by_players": {}}
    for players in player_counts:
        entry = {}
        for direction in _directions(players):
            rewards, wins, model_wins, lengths, truncated = [], 0, 0, [], 0
            for count, kind, seats, returns, inner in finished:
                if count != players or kind != direction:
                    continue
                rewards.append(float(np.mean([returns[seat] for seat in seats])))
                engine = inner.engine
                wins += engine.winner is not None
                model_wins += engine.winner in seats
                lengths.append(len(inner.oracle))
                truncated += engine.truncated
            n = len(rewards)
            entry[direction] = {
                "games": n,
                "model_relative_reward": float(np.mean(rewards)),
                "stderr": float(np.std(rewards) / np.sqrt(max(n - 1, 1))),
                "model_win_share": model_wins / max(wins, 1),
                "rounds_with_winner": wins / n,
                "events_per_game": float(np.mean(lengths)),
                "truncated": truncated / n,
            }
        entry["both_directions"] = float(
            np.mean([entry[d]["model_relative_reward"] for d in _directions(players)])
        )
        result["by_players"][str(players)] = entry
    result["model_relative_reward"] = float(
        np.mean([entry["both_directions"] for entry in result["by_players"].values()])
    )
    return result


def evaluate_command(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    actor, iteration = load_actor(args.model, device)
    players = tuple(int(p) for p in args.players.split(","))
    result = evaluate(
        actor,
        opponent=args.opponent,
        player_counts=players,
        hand_size=args.hand_size,
        deals=args.deals,
        seed=args.seed,
        device=device,
    )
    result.update({"model": str(args.model), "iteration": iteration, "opponent": args.opponent, "time": timestamp()})
    print(f"{args.model} (iteration {iteration}) vs {args.opponent}: "
          f"relative reward {result['model_relative_reward']:+.2f} points per seat")
    for count, entry in result["by_players"].items():
        alone = entry["model_alone"]
        line = (
            f"  {count}p: model alone {alone['model_relative_reward']:+7.2f} +- {alone['stderr']:.2f}"
            f" (wins {alone['model_win_share']:.2f})"
        )
        crowd = entry.get("opponent_alone")
        if crowd is not None:
            line += f"   opponent alone {crowd['model_relative_reward']:+7.2f} +- {crowd['stderr']:.2f}"
        print(line + f"   events/game {alone['events_per_game']:.0f}")
    if args.output:
        atomic_write_json(args.output, result)
    return 0
