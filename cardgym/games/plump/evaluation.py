"""Paired evaluation of a model against a fixed opponent on a fixed deal bank.

For every (players, hand size) shape the bank holds ``deals`` seeded deals.
Each deal is played P x P times: the model takes every one of the P hands in
turn, from every one of the P bidding positions, with the opponent policy in
the other seats. Holding deals and positions fixed pairs the comparison
across checkpoints, so differences are not swamped by deal luck.

The headline number is the model seat's relative reward (its score minus the
mean opponent score), averaged uniformly over shapes, with a bootstrap
confidence interval over deals.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np

from cardgym.core.io import atomic_write_json, timestamp

from .cards import make_deck
from .env import PlumpEnv
from .heuristic import HeuristicPolicy
from .policy import PlumpPolicy
from .rounds import RoundSpec, round_game_config, rules_fingerprint
from .rules import relative_rewards


@dataclass
class Scenario:
    env: PlumpEnv
    shape: tuple[int, int]
    deal: int
    focal: int


def deal_bank(
    player_counts: tuple[int, ...], hand_sizes: tuple[int, ...], deals: int, seed: int
) -> list[tuple[tuple[int, int], int, list[list]]]:
    rng = random.Random(seed)
    bank = []
    for players in player_counts:
        for hand_size in hand_sizes:
            RoundSpec(players, hand_size).validate()
            for deal in range(deals):
                deck = make_deck()
                rng.shuffle(deck)
                hands = [deck[seat * hand_size : (seat + 1) * hand_size] for seat in range(players)]
                bank.append(((players, hand_size), deal, hands))
    return bank


def scenarios(bank) -> list[Scenario]:
    result = []
    for shape, deal, hands in bank:
        players, hand_size = shape
        for focal in range(players):
            for start in range(players):
                env = PlumpEnv(
                    round_game_config(
                        RoundSpec(players, hand_size),
                        bidding_start_player=start,
                        manual_hands={seat: list(hand) for seat, hand in enumerate(hands)},
                    ),
                    seed=0,
                )
                env.emit_observations = False
                env.reset()
                result.append(Scenario(env=env, shape=shape, deal=deal, focal=focal))
    return result


def play(scenarios_: list[Scenario], model, opponent, *, batch_size: int, seed: int) -> None:
    """Advance every scenario to the end, batching each policy's decisions."""

    rngs = [random.Random(seed * 1_000_003 + index) for index in range(len(scenarios_))]
    live = list(range(len(scenarios_)))
    while live:
        for policy, focal_turn in ((model, True), (opponent, False)):
            ready = [
                index
                for index in live
                if not scenarios_[index].env.is_done()
                and (scenarios_[index].env.current_player() == scenarios_[index].focal) == focal_turn
            ]
            for start in range(0, len(ready), batch_size):
                chunk = ready[start : start + batch_size]
                envs = [scenarios_[index].env for index in chunk]
                if isinstance(policy, PlumpPolicy):
                    actions = policy.act_many(envs, [rngs[index] for index in chunk])
                else:
                    actions = policy.act_many(envs)
                for env, action in zip(envs, actions):
                    env.step(action)
        live = [index for index in live if not scenarios_[index].env.is_done()]


def report(scenarios_: list[Scenario], *, bootstrap: int = 2000, seed: int = 0) -> dict:
    by_shape: dict[tuple[int, int], dict[int, list[tuple[float, float]]]] = {}
    for scenario in scenarios_:
        round_state = scenario.env.state.current_round
        reward = relative_rewards(round_state.round_scores)[scenario.focal]
        bid = next(b.value for b in round_state.bids if b.player == scenario.focal)
        hit = float(round_state.tricks_won[scenario.focal] == bid)
        by_shape.setdefault(scenario.shape, {}).setdefault(scenario.deal, []).append((reward, hit))
    shapes = []
    deal_means = []
    for shape in sorted(by_shape):
        per_deal = np.asarray([np.mean(values, axis=0) for values in by_shape[shape].values()])
        deal_means.append(per_deal[:, 0])
        shapes.append(
            {
                "players": shape[0],
                "hand_size": shape[1],
                "rounds": sum(len(values) for values in by_shape[shape].values()),
                "relative_reward": float(per_deal[:, 0].mean()),
                "bid_hit_rate": float(per_deal[:, 1].mean()),
            }
        )
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(bootstrap):
        samples.append(
            np.mean([means[rng.integers(0, len(means), len(means))].mean() for means in deal_means])
        )
    return {
        "relative_reward": float(np.mean([s["relative_reward"] for s in shapes])),
        "relative_reward_ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "bid_hit_rate": float(np.mean([s["bid_hit_rate"] for s in shapes])),
        "rounds": len(scenarios_),
        "shapes": shapes,
    }


def evaluate(
    model: PlumpPolicy,
    opponent,
    *,
    player_counts: tuple[int, ...] = (3, 4, 5),
    hand_sizes: tuple[int, ...] = tuple(range(3, 11)),
    deals: int = 32,
    seed: int = 1234,
    batch_size: int = 256,
) -> dict:
    played = scenarios(deal_bank(player_counts, hand_sizes, deals, seed))
    play(played, model, opponent, batch_size=batch_size, seed=seed)
    result = report(played, seed=seed)
    result["protocol"] = {
        "deals_per_shape": deals,
        "seed": seed,
        "player_counts": list(player_counts),
        "hand_sizes": list(hand_sizes),
        "model_mode": model.mode,
        "rules": rules_fingerprint(),
    }
    return result


def evaluate_command(args: argparse.Namespace) -> int:
    model = PlumpPolicy.load(args.model, device=args.device, mode=args.mode)
    if args.opponent == "heuristic":
        opponent = HeuristicPolicy()
    else:
        opponent = PlumpPolicy.load(args.opponent, device=args.device, mode=args.mode)
    result = evaluate(model, opponent, deals=args.deals, seed=args.seed, batch_size=args.batch_size)
    result["model"] = str(args.model)
    result["opponent"] = str(args.opponent)
    result["evaluated_at"] = timestamp()
    print(f"{'players':>7} {'cards':>5} {'rounds':>7} {'reward':>8} {'bid hit':>8}")
    for shape in result["shapes"]:
        print(
            f"{shape['players']:7d} {shape['hand_size']:5d} {shape['rounds']:7d} "
            f"{shape['relative_reward']:8.3f} {shape['bid_hit_rate']:8.3f}"
        )
    low, high = result["relative_reward_ci95"]
    print(
        f"relative reward vs {args.opponent}: {result['relative_reward']:+.3f} "
        f"(95% CI {low:+.3f} .. {high:+.3f}), bid hit {result['bid_hit_rate']:.3f}, "
        f"{result['rounds']} rounds"
    )
    if args.output:
        atomic_write_json(args.output, result)
    return 0

