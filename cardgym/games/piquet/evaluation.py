"""Duplicate evaluation of a model against a rule-based opponent or another model.

Each seeded deal (or partie) is played twice with the seats swapped, so both
sides hold both hands, as Elder and as Younger. The headline is the model's
mean return per game (points per deal, or the Rubicon payoff), with a
standard error over the duplicate pairs.

Decoding reuses the continuous collector: rule-based seats act inside a
game's ``step``, and only the model seat's tokens reach the decoder. Against
another model (``--against``) every seat is decoded by its own model
(cardgym.rl.match). A checkpoint from before the partie fields loads blind
to them (PiquetEmbedding), so it plays every deal of a partie as a single
deal: the score-blind baseline a partie-trained model should beat.
"""

from __future__ import annotations

import argparse
import functools
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from cardgym.core.io import atomic_write_json, timestamp
from cardgym.rl.checkpoint import load_weights
from cardgym.rl.continuous import ContinuousCollector, Lane
from cardgym.rl.match import MatchCollector
from cardgym.rl.selfplay import Decision, SelfPlayGame

from .engine import PiquetEngine
from .heuristic import HeuristicPolicy, RandomPolicy
from .model import PiquetActor, PiquetModelConfig
from .selfplay import EMPTY, PiquetSelfPlay
from .tokens import DEAL_TOKENS


class MixedGame(SelfPlayGame):
    """A game in which only ``model_seat`` asks the collector for decisions."""

    def __init__(self, inner: PiquetSelfPlay, model_seat: int, opponent, pair: int) -> None:
        self.inner = inner
        self.num_players = 2
        self.rng = inner.rng
        self.model_seat = model_seat
        self.opponent = opponent
        self.pair = pair
        self._starts: list[int | None] | None = None
        self._prefix = self._advance(inner.prefix_tokens())
        if self._starts is not None:
            raise AssertionError("The model seat must decide before the first deal ends.")

    def _advance(self, blocks: list[np.ndarray]) -> list[np.ndarray]:
        """The model seat's tokens until it decides again; notes where its last fresh context begins.

        ``blocks`` come from the inner game's latest flush (its prefix, or
        the model's own step), so its context starts describe them.
        """

        engine = self.inner.engine
        seat = self.model_seat
        pending = [blocks[seat]]
        start = (self.inner.context_starts() or [None, None])[seat]
        offset = len(blocks[seat])
        while not engine.done and engine.current != seat:
            block = self.inner.step(self.opponent.act(engine))[seat]
            restart = (self.inner.context_starts() or [None, None])[seat]
            if restart is not None:
                start = offset + restart
            pending.append(block)
            offset += len(block)
        out = [EMPTY, EMPTY]
        out[seat] = np.concatenate(pending)
        self._starts = None if start is None else [start if s == seat else None for s in (0, 1)]
        return out

    def lifetime(self) -> int:
        return self.inner.lifetime()

    def prefix_tokens(self) -> list[np.ndarray]:
        return self._prefix

    def decision(self) -> Decision:
        return self.inner.decision()

    def step(self, action: int) -> list[np.ndarray]:
        return self._advance(self.inner.step(action))

    def context_starts(self) -> list[int | None] | None:
        return self._starts

    def done(self) -> bool:
        return self.inner.done()

    def returns(self) -> np.ndarray:
        return self.inner.returns()


def load_actor(path: Path, device: torch.device) -> tuple[PiquetActor, int]:
    from cardgym.core.config import parse_section

    weights = load_weights(path, "piquet")
    actor = PiquetActor(parse_section(PiquetModelConfig, weights["model"], "model"))
    actor.load_state_dict(weights["actor"])
    return actor.to(device).eval(), weights["iteration"]


def _next_game(games: deque, rng: random.Random):
    return games.popleft() if games else None


def evaluate(
    actor: PiquetActor,
    *,
    opponent: str,
    deals: int,
    seed: int,
    device: torch.device,
    mode: str = "deal",
    precision: str = "fp32",
    against: PiquetActor | None = None,
) -> dict:
    """Duplicate games of ``actor`` against a rule-based ``opponent``, or against the model ``against``."""

    rng = random.Random(seed)
    games: deque[SelfPlayGame] = deque()
    for pair in range(deals):
        deal_seed = rng.getrandbits(48)
        for model_seat in (0, 1):
            inner = PiquetSelfPlay(deal_seed, mode=mode)
            if against is None:
                policy = HeuristicPolicy() if opponent == "heuristic" else RandomPolicy(deal_seed ^ model_seat)
                games.append(MixedGame(inner, model_seat, policy, pair))
            else:
                inner.seat_models = (0, 1) if model_seat == 0 else (1, 0)
                inner.model_seat, inner.pair = model_seat, pair
                games.append(inner)
    lane = Lane(name=mode, players=2, in_flight=len(games), quota=None, deal=functools.partial(_next_game, games))
    options = dict(device=device, precision=precision, kv_dtype=precision, max_length=DEAL_TOKENS)
    collector = (
        ContinuousCollector(actor, **options) if against is None else MatchCollector([actor, against], **options)
    )
    trajectories = collector.collect([lane], rng)
    collector.reset()
    return report([(_engine(t.game), t.game.model_seat, t.game.pair) for t in trajectories])


def _engine(game: SelfPlayGame) -> PiquetEngine:
    return game.inner.engine if isinstance(game, MixedGame) else game.engine


def report(games: list[tuple[PiquetEngine, int, int]]) -> dict:
    """Statistics of (engine, model seat, duplicate pair) games, from the model's side."""

    pairs: dict[int, list[float]] = {}
    elder, younger, results = [], [], []
    rubicon = {"model_rubiconed": 0, "opponent_rubiconed": 0}
    for engine, seat, pair in games:
        value = float(engine.returns()[seat])
        pairs.setdefault(pair, []).append(value)
        for result in engine.results:
            margin = result.points[seat] - result.points[1 - seat]
            (elder if result.elder == seat else younger).append(margin)
            results.append((result, seat))
        if engine.rubicon and value != 0:
            loser = seat if value < 0 else 1 - seat
            if engine.totals[loser] < 100:
                rubicon["model_rubiconed" if loser == seat else "opponent_rubiconed"] += 1
    per_pair = np.array([np.mean(values) for values in pairs.values()])
    n = len(per_pair)
    deals = len(results)
    stats = {
        "games": len(games),
        "model_return": float(per_pair.mean()),
        "stderr": float(per_pair.std() / np.sqrt(max(n - 1, 1))),
        "win_rate": float(np.mean([engine.returns()[seat] > 0 for engine, seat, _ in games])),
        "elder_margin": float(np.mean(elder)),
        "younger_margin": float(np.mean(younger)),
        "model_repique": sum(r.repique == s for r, s in results) / deals,
        "model_pique": sum(r.pique == s for r, s in results) / deals,
        "model_capot": sum(r.capot == s for r, s in results) / deals,
        "opponent_repique": sum(r.repique == 1 - s for r, s in results) / deals,
        "opponent_pique": sum(r.pique == 1 - s for r, s in results) / deals,
        "opponent_capot": sum(r.capot == 1 - s for r, s in results) / deals,
    }
    if games and games[0][0].rubicon:
        stats["points_margin"] = float(np.mean([e.totals[s] - e.totals[1 - s] for e, s, _ in games]))
        stats.update({key: count / len(games) for key, count in rubicon.items()})
    return stats


def evaluate_command(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    actor, iteration = load_actor(args.model, device)
    against = None
    if args.against is not None:
        against, against_iteration = load_actor(args.against, device)
    result = evaluate(
        actor, opponent=args.opponent, deals=args.deals, seed=args.seed, device=device, mode=args.mode,
        against=against,
    )
    opponent = args.opponent if against is None else f"{args.against} (iteration {against_iteration})"
    result.update({"model": str(args.model), "iteration": iteration, "opponent": opponent,
                   "mode": args.mode, "time": timestamp()})
    unit = "points per deal" if args.mode == "deal" else "per partie (Rubicon payoff)"
    print(f"{args.model} (iteration {iteration}) vs {opponent}: "
          f"{result['model_return']:+.2f} +- {result['stderr']:.2f} {unit}, wins {result['win_rate']:.3f}")
    print(f"  as elder {result['elder_margin']:+.2f}, as younger {result['younger_margin']:+.2f} points per deal")
    if "points_margin" in result:
        print(f"  point margin per partie {result['points_margin']:+.1f}; rubiconed: "
              f"model {result['model_rubiconed']:.3f}, opponent {result['opponent_rubiconed']:.3f}")
    print(f"  repique {result['model_repique']:.3f} vs {result['opponent_repique']:.3f}, "
          f"pique {result['model_pique']:.3f} vs {result['opponent_pique']:.3f}, "
          f"capot {result['model_capot']:.3f} vs {result['opponent_capot']:.3f}")
    if args.output:
        atomic_write_json(args.output, result)
    return 0
