"""Inference: a trained plump actor acting on live environments.

Each decision re-encodes the observer's stream from the event log with one
full causal forward (no cache), batched across environments. That is the
simplest correct path for evaluation and interactive play; self-play uses the
KV-cached collector instead.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from cardgym.core.config import parse_section
from cardgym.rl.checkpoint import load_weights
from cardgym.rl.ppo import masked_probabilities, sample_rows

from .env import PlumpEnv
from .model import PlumpActor, PlumpModelConfig
from .state import BidAction, Phase, PlayCardAction
from .tokens import NEXT_BID, NEXT_PLAY, NUM_CARDS, TOKEN_WIDTH, card_from_id, card_id, seat_tokens


class PlumpPolicy:
    def __init__(
        self,
        actor: PlumpActor,
        *,
        device: str | torch.device = "cpu",
        mode: str = "argmax",
        name: str = "model",
    ) -> None:
        if mode not in ("argmax", "sample"):
            raise ValueError("mode must be 'argmax' or 'sample'.")
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.schema = actor.schema
        self.mode = mode
        self.name = name

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu", mode: str = "argmax") -> "PlumpPolicy":
        """From a release directory, a release actor.pt, or a training checkpoint."""

        path = Path(path)
        weights = load_weights(path, "plump")
        actor = PlumpActor(parse_section(PlumpModelConfig, weights["model"], "model"))
        actor.load_state_dict(weights["actor"])
        return cls(actor, device=device, mode=mode, name=path.name)

    def stream(self, env: PlumpEnv, observer: int) -> np.ndarray:
        """``observer``'s stream ending at the readout for the pending decision."""

        round_state = env.state.current_round
        return seat_tokens(
            self.schema,
            env.state.event_log,
            observer,
            env.config.num_players,
            round_state.hand_size,
            round_state.initial_hands[observer],
            round_state.bidding_start_player,
            pending_actor=env.current_player(),
            pending_phase=NEXT_BID if env.phase() == Phase.BIDDING else NEXT_PLAY,
        )

    @torch.inference_mode()
    def hidden(self, envs: list[PlumpEnv], observers: list[int]) -> torch.Tensor:
        """Final hidden state of each (env, observer) stream, [B, d]."""

        streams = [self.stream(env, observer) for env, observer in zip(envs, observers)]
        lengths = [stream.shape[0] for stream in streams]
        batch = np.zeros((len(streams), max(lengths), TOKEN_WIDTH), dtype=np.int64)
        for row, stream in enumerate(streams):
            batch[row, : stream.shape[0]] = stream
        hidden = self.actor.forward_hidden(torch.from_numpy(batch).to(self.device))
        rows = torch.arange(len(streams), device=self.device)
        return hidden[rows, torch.tensor(lengths, device=self.device) - 1]

    @torch.inference_mode()
    def action_probabilities(self, envs: list[PlumpEnv]) -> list[np.ndarray]:
        """Masked policy over the head's actions (11 bids or 52 cards) per env."""

        hidden = self.hidden(envs, [env.current_player() for env in envs])
        result: list[np.ndarray | None] = [None] * len(envs)
        for head in ("bid", "play"):
            rows = [i for i, env in enumerate(envs) if (env.phase() == Phase.BIDDING) == (head == "bid")]
            if not rows:
                continue
            logits = self.actor.policy_logits(hidden[rows], head).float().cpu().numpy()
            mask = np.zeros_like(logits, dtype=bool)
            for row, index in enumerate(rows):
                env = envs[index]
                legal = env.legal_bid_values() if head == "bid" else [card_id(c) for c in env.legal_card_values()]
                mask[row, legal] = True
            probabilities = masked_probabilities(logits, mask)
            for row, index in enumerate(rows):
                result[index] = probabilities[row]
        return result

    def act_many(
        self, envs: list[PlumpEnv], rngs: list[random.Random] | None = None
    ) -> list[BidAction | PlayCardAction]:
        if not envs:
            return []
        probabilities = self.action_probabilities(envs)
        actions = []
        for index, (env, probs) in enumerate(zip(envs, probabilities)):
            if self.mode == "argmax":
                choice = int(probs.argmax())
            else:
                uniform = (rngs[index] if rngs else random).random()
                choice = int(sample_rows(probs[None], probs[None] > 0, np.asarray([uniform]))[0])
            player = env.current_player()
            if env.phase() == Phase.BIDDING:
                actions.append(BidAction(player, choice))
            else:
                if choice >= NUM_CARDS:
                    raise AssertionError("Card index out of range.")
                actions.append(PlayCardAction(player, card_from_id(choice)))
        return actions

    def act(self, env: PlumpEnv) -> BidAction | PlayCardAction:
        return self.act_many([env])[0]
