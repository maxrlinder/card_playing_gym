"""Local browser app: play plump against a trained model (or random bots).

Seat 0 is the human. With a model loaded, the page also shows every seat's
model beliefs (opponent suit presence, final-trick forecasts) and, on the
human's turn, the model's own action probabilities for the human's hand.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch

from ..cards import Card, Rank, Suit, card_str
from ..env import PlumpEnv
from ..policy import PlumpPolicy
from ..rounds import descending_ascending_schedule
from ..state import BidAction, GameConfig, IllegalActionError, Phase, PlayCardAction
from ..tokens import NUM_SUITS, SUITS, card_id

HUMAN = 0
DISPLAY_SUIT_ORDER = {Suit.HEARTS: 0, Suit.SPADES: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}
STATIC = Path(__file__).with_name("static")


@dataclass
class GuiGame:
    env: PlumpEnv
    rng: random.Random
    messages: list[str]
    mode: str
    show_probabilities: bool
    announced_rounds: set[int]


class ModelView:
    """The loaded policy plus the belief readouts the page displays."""

    def __init__(self, path: Path, device: str) -> None:
        self.path = path
        self.policy = PlumpPolicy.load(path, device=device, mode="argmax")

    def act(self, env: PlumpEnv):
        return self.policy.act(env)

    @torch.inference_mode()
    def predictions(self, env: PlumpEnv, *, with_actions: bool) -> tuple[dict[int, dict], dict | None]:
        players = env.config.num_players
        actor = self.policy.actor
        schema = actor.schema
        hidden = self.policy.hidden([env] * players, list(range(players)))
        aux = actor.auxiliary_logits(hidden, ("suit", "trick"))
        suit = torch.sigmoid(aux["suit"].float().view(players, schema.opponents, NUM_SUITS))
        trick = aux["trick"].float().view(players, schema.max_players, schema.bid_count)
        round_state = env.state.current_round
        unresolved = round_state.hand_size - sum(t.winner is not None for t in round_state.tricks)
        result: dict[int, dict] = {}
        for observer in range(players):
            rows = []
            for rel in range(players):
                seat = (observer + rel) % players
                forecast = _trick_forecast(trick[observer, rel], round_state.tricks_won.get(seat, 0), unresolved)
                rows.append(
                    {
                        "player": seat,
                        "expected_final_tricks": forecast["expected"],
                        "top_final_tricks": forecast["top"],
                        "suit_presence": (
                            {s.value: float(suit[observer, rel - 1, i]) for i, s in enumerate(SUITS)}
                            if rel
                            else None
                        ),
                    }
                )
            result[observer] = {"observer": observer, "source": "checkpoint", "rows": rows}
        actions = None
        if with_actions and env.state.current_player == HUMAN and env.phase() in (Phase.BIDDING, Phase.PLAYING):
            actions = self._human_actions(env, hidden[HUMAN])
        return result, actions

    def _human_actions(self, env: PlumpEnv, hidden: torch.Tensor) -> dict:
        actor = self.policy.actor
        if env.phase() == Phase.BIDDING:
            legal = env.legal_bid_values()
            logits = actor.policy_logits(hidden[None], "bid")[0, legal]
            keys = [{"bid": bid} for bid in legal]
        else:
            cards = env.legal_card_values()
            logits = actor.policy_logits(hidden[None], "play")[0, [card_id(c) for c in cards]]
            keys = [{"card_key": card_key(card)} for card in cards]
        probabilities = torch.softmax(logits.float(), dim=-1).tolist()
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "phase": env.phase().value,
            "actions": [
                {**key, "probability": float(p), "is_best": index == best}
                for index, (key, p) in enumerate(zip(keys, probabilities))
            ],
        }


def _trick_forecast(logits: torch.Tensor, tricks_won: int, unresolved: int) -> dict[str, Any]:
    """Final-trick distribution restricted to totals still reachable.

    Training masks the head to feasible totals; logits outside that interval
    received no loss and must not be shown as beliefs.
    """

    counts = torch.arange(logits.shape[-1])
    feasible = (counts >= tricks_won) & (counts <= min(tricks_won + unresolved, logits.shape[-1] - 1))
    probabilities = torch.softmax(logits.masked_fill(~feasible, float("-inf")), dim=-1)
    top_probabilities, top_counts = probabilities.topk(min(3, int(feasible.sum())))
    return {
        "expected": float((counts * probabilities).sum()),
        "top": [
            {"tricks": int(count), "probability": float(probability)}
            for probability, count in zip(top_probabilities, top_counts)
        ],
    }


class GuiController:
    def __init__(self, model_path: Path | None = None, device: str = "cpu") -> None:
        self.game: GuiGame | None = None
        self.model = ModelView(model_path, device) if model_path is not None else None

    def new_game(
        self,
        opponents: int,
        hand_size: int,
        human_bid_position: int,
        seed: int | None = None,
        *,
        mode: str = "round",
        min_hand_size: int = 3,
        max_hand_size: int = 10,
        show_probabilities: bool = True,
    ) -> dict[str, Any]:
        if not 2 <= opponents <= 4:
            raise ValueError("Choose 2 to 4 opponents.")
        players = opponents + 1
        if not 1 <= human_bid_position <= players:
            raise ValueError(f"Your bid order must be 1 to {players}.")
        if mode == "round":
            if not 3 <= hand_size <= 10:
                raise ValueError("Choose 3 to 10 cards.")
            hand_sizes = [hand_size]
        elif mode == "game":
            if not 5 <= max_hand_size <= 10 or not 3 <= min_hand_size < max_hand_size:
                raise ValueError("Choose a highest round of 5-10 cards and a lower lowest round of at least 3.")
            hand_sizes = descending_ascending_schedule(min_cards=min_hand_size, max_cards=max_hand_size)
        else:
            raise ValueError("Mode must be 'round' or 'game'.")
        start = (HUMAN - (human_bid_position - 1)) % players
        config = GameConfig(
            num_players=players,
            hand_sizes=hand_sizes,
            bidding_start_players=[(start + index) % players for index in range(len(hand_sizes))],
            auto_advance_rounds=False,
        )
        env = PlumpEnv(config, seed=seed)
        env.reset()
        opening = (
            f"New round: {players} players, {hand_size} cards."
            if mode == "round"
            else f"New full game: {players} players, {len(hand_sizes)} rounds from "
            f"{max_hand_size} down to {min_hand_size} and back."
        )
        self.game = GuiGame(
            env=env,
            rng=random.Random(seed),
            messages=[opening, f"You bid {ordinal(human_bid_position)} in the first round."],
            mode=mode,
            show_probabilities=show_probabilities,
            announced_rounds=set(),
        )
        self._advance_bidding_bots()
        return self.view()

    def bid(self, value: int) -> dict[str, Any]:
        game = self._human_turn(Phase.BIDDING)
        game.env.step(BidAction(HUMAN, value))
        game.messages.append(f"You bid {value}.")
        self._advance_bidding_bots()
        return self.view()

    def play(self, suit: str, rank: int) -> dict[str, Any]:
        game = self._human_turn(Phase.PLAYING)
        card = Card(Suit(suit), Rank(rank))
        game.env.step(PlayCardAction(HUMAN, card))
        game.messages.append(f"You played {card_str(card)}.")
        self._announce_completed_round()
        return self.view()

    def next_round(self) -> dict[str, Any]:
        game = self._require_game()
        if game.env.phase() != Phase.ROUND_OVER:
            raise IllegalActionError("The next round can only start after a completed round.")
        game.env.start_next_round()
        round_state = game.env.state.current_round
        game.messages.append(f"Round {round_state.round_index + 1} started with {round_state.hand_size} cards.")
        self._advance_bidding_bots()
        return self.view()

    def set_probability_visibility(self, visible: bool) -> dict[str, Any]:
        self._require_game().show_probabilities = visible
        return self.view()

    def advance_bot(self) -> dict[str, Any]:
        game = self._require_game()
        if game.env.phase() in (Phase.BIDDING, Phase.PLAYING) and game.env.current_player() != HUMAN:
            self._advance_one_bot()
        return self.view()

    def view(self) -> dict[str, Any]:
        game = self._require_game()
        env = game.env
        state = env.state
        round_state = state.current_round if state.rounds else None
        human = env.get_observation(HUMAN) if round_state is not None else None
        legal_cards = {card_key(card) for card in human.legal_cards} if human else set()
        predictions: dict[int, dict] = {}
        actions = None
        if self.model is not None and round_state is not None and state.phase in (Phase.BIDDING, Phase.PLAYING):
            predictions, actions = self.model.predictions(env, with_actions=game.show_probabilities)
        bids = {bid.player: bid.value for bid in round_state.bids} if round_state else {}
        players = [
            {
                "id": player,
                "name": player_name(player),
                "is_human": player == HUMAN,
                "is_current": state.current_player == player,
                "cards": len(round_state.current_hands[player]) if round_state else 0,
                "bid": bids.get(player),
                "tricks": round_state.tricks_won.get(player, 0) if round_state else 0,
                "score": state.cumulative_scores.get(player, 0),
                "prediction": predictions.get(player),
            }
            for player in range(env.config.num_players)
        ]
        completed_tricks = (
            [serialize_trick(t) for t in round_state.tricks if t.winner is not None] if round_state else []
        )
        completed_rounds = [
            {
                "round_number": done.round_index + 1,
                "hand_size": done.hand_size,
                "scores": done.round_scores,
                "cumulative_scores": done.cumulative_scores_after_round,
            }
            for done in state.rounds
            if done.round_scores
        ]
        winners: list[int] = []
        if env.is_done() and state.cumulative_scores:
            best = max(state.cumulative_scores.values())
            winners = [player for player, score in state.cumulative_scores.items() if score == best]
        return {
            "ok": True,
            "mode": game.mode,
            "show_probabilities": game.show_probabilities,
            "phase": state.phase.value,
            "done": env.is_done(),
            "round_over": state.phase == Phase.ROUND_OVER,
            "game_over": state.phase == Phase.GAME_OVER,
            "current_player": state.current_player,
            "human_turn": state.current_player == HUMAN and state.phase in (Phase.BIDDING, Phase.PLAYING),
            "round_number": round_state.round_index + 1 if round_state else 0,
            "total_rounds": len(env.config.hand_sizes),
            "rounds_remaining": max(len(env.config.hand_sizes) - len(completed_rounds), 0),
            "schedule": list(env.config.hand_sizes),
            "hand_size": round_state.hand_size if round_state else None,
            "trump": None,
            "trump_label": "No trump",
            "bidding_order": round_state.bidding_order if round_state else [],
            "play_start_player": round_state.play_start_player if round_state else None,
            "players": players,
            "my_hand": [
                serialize_card(card, legal=card_key(card) in legal_cards)
                for card in sorted(human.my_hand, key=lambda c: (DISPLAY_SUIT_ORDER[c.suit], int(c.rank)))
            ]
            if human
            else [],
            "legal_bids": human.legal_bids if human else [],
            "bids": [{"player": b.player, "value": b.value, "position": b.position} for b in round_state.bids]
            if round_state
            else [],
            "current_trick": serialize_trick(round_state.tricks[-1]) if round_state and round_state.tricks else None,
            "completed_tricks": completed_tricks,
            "last_trick": completed_tricks[-1] if completed_tricks else None,
            "round_scores": round_state.round_scores if round_state else {},
            "completed_rounds": completed_rounds,
            "winner_ids": winners,
            "model_action_probabilities": actions,
            "messages": game.messages[-12:],
            "event_log": serialize_events(state.event_log[-40:]),
            "model_checkpoint": str(self.model.path) if self.model is not None else None,
        }

    def _advance_bidding_bots(self) -> None:
        env = self._require_game().env
        while not env.is_done() and env.phase() == Phase.BIDDING and env.current_player() != HUMAN:
            self._advance_one_bot()

    def _advance_one_bot(self) -> None:
        game = self._require_game()
        if game.env.is_done() or game.env.current_player() == HUMAN:
            return
        action = (
            self.model.act(game.env) if self.model is not None else game.rng.choice(game.env.legal_actions())
        )
        game.env.step(action)
        if isinstance(action, BidAction):
            game.messages.append(f"Model {action.player} bid {action.bid}.")
        else:
            game.messages.append(f"Model {action.player} played {card_str(action.card)}.")
        self._announce_completed_round()

    def _announce_completed_round(self) -> None:
        game = self._require_game()
        round_state = game.env.state.current_round
        if not round_state.round_scores or round_state.round_index in game.announced_rounds:
            return
        game.announced_rounds.add(round_state.round_index)
        points = ", ".join(f"{player_name(p)} +{s}" for p, s in sorted(round_state.round_scores.items()))
        game.messages.append(f"Round {round_state.round_index + 1} over. Points: {points}.")
        if game.env.is_done():
            best = max(game.env.state.cumulative_scores.values())
            names = [player_name(p) for p, s in game.env.state.cumulative_scores.items() if s == best]
            game.messages.append(f"Game over. {' and '.join(names)} won with {best} points.")

    def _require_game(self) -> GuiGame:
        if self.game is None:
            raise RuntimeError("Start a game first.")
        return self.game

    def _human_turn(self, phase: Phase) -> GuiGame:
        game = self._require_game()
        if game.env.phase() != phase:
            raise IllegalActionError(f"Expected phase {phase.value}, got {game.env.phase().value}.")
        if game.env.current_player() != HUMAN:
            raise IllegalActionError("It is not your turn.")
        return game


def serialize_card(card: Card, legal: bool = False) -> dict[str, Any]:
    return {
        "suit": card.suit.value,
        "rank": int(card.rank),
        "label": card_str(card),
        "key": card_key(card),
        "color": "red" if card.suit in (Suit.HEARTS, Suit.DIAMONDS) else "black",
        "legal": legal,
    }


def serialize_trick(trick) -> dict[str, Any]:
    return {
        "index": trick.trick_index,
        "leader": trick.leader,
        "led_suit": trick.led_suit.value if trick.led_suit else None,
        "winner": trick.winner,
        "plays": [
            {"player": play.player, "position": play.position, "card": serialize_card(play.card)}
            for play in trick.plays
        ],
    }


def serialize_events(events) -> list[dict[str, Any]]:
    return [
        {
            "type": event.type.value,
            "round_index": event.round_index,
            "player": event.player,
            "card": serialize_card(event.card) if event.card else None,
            "bid": event.bid,
            "trick_index": event.trick_index,
            "position_in_trick": event.position_in_trick,
        }
        for event in events
    ]


def card_key(card: Card) -> str:
    return f"{card.suit.value}:{int(card.rank)}"


def player_name(player: int) -> str:
    return "You" if player == HUMAN else f"Model {player}"


def ordinal(value: int) -> str:
    suffix = "th" if 10 <= value % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


class GuiRequestHandler(BaseHTTPRequestHandler):
    controller = GuiController()
    STATIC_FILES = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    }

    def do_GET(self) -> None:
        if self.path in self.STATIC_FILES:
            name, content_type = self.STATIC_FILES[self.path]
            data = (STATIC / name).read_bytes()
            self._send(HTTPStatus.OK, content_type, data)
        elif self.path == "/api/state":
            self._json(self.controller.view())
        else:
            self._json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
            routes = {
                "/api/new": lambda: self.controller.new_game(
                    opponents=int(payload.get("opponents", 3)),
                    hand_size=int(payload.get("hand_size", 5)),
                    human_bid_position=int(payload.get("bid_position", 1)),
                    seed=int(payload["seed"]) if payload.get("seed") not in (None, "") else None,
                    mode=str(payload.get("mode", "round")),
                    min_hand_size=int(payload.get("min_hand_size", 3)),
                    max_hand_size=int(payload.get("max_hand_size", 10)),
                    show_probabilities=bool(payload.get("show_probabilities", True)),
                ),
                "/api/bid": lambda: self.controller.bid(int(payload["bid"])),
                "/api/play": lambda: self.controller.play(str(payload["suit"]), int(payload["rank"])),
                "/api/advance": self.controller.advance_bot,
                "/api/next-round": self.controller.next_round,
                "/api/probabilities": lambda: self.controller.set_probability_visibility(
                    _boolean(payload["visible"])
                ),
            }
            if self.path not in routes:
                self._json({"ok": False, "error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            self._json(routes[self.path]())
        except (KeyError, ValueError, RuntimeError, IllegalActionError) as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def _send(self, status: HTTPStatus, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # A local development app: never serve stale HTML/JS/CSS after a restart.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def _boolean(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Probability visibility must be a boolean.")
    return value


def serve(model: Path | None, *, device: str = "cpu", host: str = "127.0.0.1", port: int = 8765) -> int:
    if model is not None and not Path(model).exists():
        raise FileNotFoundError(f"{model} does not exist (pass a release directory or checkpoint).")
    GuiRequestHandler.controller = GuiController(model, device=device)
    server = ThreadingHTTPServer((host, port), GuiRequestHandler)
    print(f"Plump GUI running at http://{host}:{port} (model: {model or 'random bots'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
