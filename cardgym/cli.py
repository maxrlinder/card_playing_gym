"""``cardgym`` command line.

    cardgym games
    cardgym <game> train RUN --config configs/<game>/<preset>.toml [--set k=v ...]
    cardgym <game> dashboard RUN
    cardgym <game> export RUN --out checkpoints/<game>/<name> [--checkpoint latest]
    cardgym <game> grow NEW --from RUN --config DEEPER.toml --fresh 4,5,10,11
    cardgym <game> <game-specific command> ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cardgym.games import GAMES, game_class


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cardgym", description="Self-play reinforcement learning gym for card games."
    )
    games = parser.add_subparsers(dest="game", required=True, metavar="{games," + ",".join(GAMES) + "}")
    listing = games.add_parser("games", help="list registered games")
    listing.set_defaults(func=lambda args: _list_games())
    for name in GAMES:
        cls = game_class(name)
        game_parser = games.add_parser(name, help=(cls.__doc__ or "").strip().splitlines()[0])
        commands = game_parser.add_subparsers(dest="command", required=True)
        _add_shared_commands(name, commands)
        cls.add_commands(commands)
    return parser


def _add_shared_commands(game: str, commands: argparse._SubParsersAction) -> None:
    from cardgym.rl.loop import add_train_arguments, train

    train_parser = commands.add_parser("train", help="create or resume a self-play PPO run")
    add_train_arguments(train_parser)
    train_parser.set_defaults(func=lambda args: train(game, args))

    dashboard = commands.add_parser("dashboard", help="render runs/<game>/<run>/dashboard.png")
    dashboard.add_argument("run")
    dashboard.set_defaults(func=lambda args: _dashboard(game, args))

    export = commands.add_parser("export", help="write a publishable release (actor/critic weights)")
    export.add_argument("run")
    export.add_argument("--checkpoint", default="latest")
    export.add_argument("--out", type=Path, required=True)
    export.set_defaults(func=lambda args: _export(game, args))

    from cardgym.rl.grow import add_grow_arguments, grow_command

    grow = commands.add_parser("grow", help="fork a run into a deeper model (fresh identity blocks)")
    add_grow_arguments(grow)
    grow.set_defaults(func=lambda args: grow_command(game, args))


def _list_games() -> int:
    for name in GAMES:
        doc = (game_class(name).__doc__ or "").strip().splitlines()[0]
        print(f"{name:10s} {doc}")
    return 0


def _dashboard(game: str, args: argparse.Namespace) -> int:
    from cardgym.core.dashboard import render_dashboard
    from cardgym.core.runs import RunDirectory
    from cardgym.rl.loop import default_panels
    from cardgym.rl.setup import resolve_config

    run = RunDirectory(game, args.run)
    config = resolve_config(run.recorded_config())
    panels = default_panels(config.game.policy_heads) + config.game.dashboard_panels()
    if not render_dashboard(run.metrics, run.dashboard, panels, title=f"{game} / {run.name}"):
        print("No metrics to plot yet.")
        return 1
    print(run.dashboard)
    return 0


def _export(game: str, args: argparse.Namespace) -> int:
    from cardgym.core.config import section_dict
    from cardgym.core.runs import RunDirectory
    from cardgym.rl.checkpoint import export_release, load_checkpoint
    from cardgym.rl.setup import resolve_config

    run = RunDirectory(game, args.run)
    checkpoint = run.resolve_checkpoint(args.checkpoint)
    payload = load_checkpoint(checkpoint)
    config = resolve_config(payload["config"])
    directory = export_release(
        args.out,
        game=game,
        iteration=int(payload["iteration"]),
        model_config=section_dict(config.game.model_config),
        actor_state=payload["trainer"]["actor"],
        critic_state=payload["trainer"]["critic"],
        entropy_state=payload["trainer"].get("entropy"),
        source={"run": run.name, "checkpoint": checkpoint.name},
    )
    print(directory)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
