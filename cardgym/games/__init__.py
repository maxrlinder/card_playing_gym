"""Game registry. Games load lazily so one game's imports never slow another."""

from __future__ import annotations

import importlib

GAMES: dict[str, str] = {
    "plump": "cardgym.games.plump.game:PlumpGame",
    "rummy": "cardgym.games.rummy.game:RummyGame",
}


def game_class(name: str):
    try:
        target = GAMES[name]
    except KeyError:
        raise ValueError(f"Unknown game {name!r}; registered: {sorted(GAMES)}") from None
    module, attribute = target.split(":")
    return getattr(importlib.import_module(module), attribute)
