"""Streams carry exactly the viewer's knowledge, labels line up, and rollout equals replay."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from cardgym.games.piquet.cards import NUM_CARDS, cards_of, mask_array
from cardgym.games.piquet.engine import (
    ELDER_MAX,
    EV_ANNOUNCE,
    EV_CARTE_BLANCHE,
    EV_DEAL,
    EV_DEAL_END,
    EV_EXCHANGE,
    EV_PLAY,
    EV_RESOLVE,
    EV_SELECT,
    HEADS,
)
from cardgym.games.piquet.game import PiquetGame, PiquetGameConfig
from cardgym.games.piquet.heuristic import HeuristicPolicy, RandomPolicy
from cardgym.games.piquet.model import PiquetModelConfig
from cardgym.games.piquet.selfplay import PiquetSelfPlay
from cardgym.games.piquet.tokens import (
    BELIEF_DISCARD,
    BELIEF_HAND,
    BELIEF_TALON,
    CARD_START,
    CARD_VOCAB,
    DEAL_TOKENS,
    EVENT_KINDS,
    F_ACTOR,
    F_CARD,
    F_DEAL,
    F_HEAD,
    F_KIND,
    F_LEAD,
    F_MODE,
    F_MY_POINTS,
    F_MY_TOTAL,
    F_OPP_POINTS,
    F_OPP_TOTAL,
    KIND_DECISION,
    LEAD_VOCAB,
    PARTIE_FIELDS,
    TOTAL_VOCAB,
    UNCERTAIN,
    V_DISCARDED,
    V_MINE,
    V_MINE_LEGAL,
    V_OPP,
    V_OPP_MAYBE,
    V_PLAYED_ME,
    V_PLAYED_OPP,
    V_SEEN_TALON,
    V_SELECTED,
    V_TRICK_ME,
    V_TRICK_OPP,
    V_UNKNOWN,
    lead_code,
    total_code,
)
from cardgym.games.piquet.training import IGNORE, PiquetLossConfig, training_sequences
from cardgym.rl.continuous import ContinuousCollector
from cardgym.rl.ppo import masked_probabilities
from tests.conftest import randomize


def drive(game: PiquetSelfPlay, policy) -> tuple[list[np.ndarray], list[tuple]]:
    """Play a game by hand: each seat's stream and every decision
    (seat, head, position, legal, action, engine events so far). The stream
    positions where the game restarts the context land in ``game.observed_starts``."""

    streams = [[block] for block in game.prefix_tokens()]
    assert game.context_starts() is None
    game.observed_starts = [[], []]
    decisions = []
    while not game.done():
        decision = game.decision()
        position = sum(len(block) for block in streams[decision.seat]) - 1
        action = policy.act(game.engine)
        decisions.append(
            (decision.seat, decision.head, position, decision.legal.copy(), action, len(game.engine.events))
        )
        blocks = game.step(action)
        for seat, start in enumerate(game.context_starts() or [None, None]):
            if start is not None:
                game.observed_starts[seat].append(sum(len(block) for block in streams[seat]) + start)
        for seat, block in enumerate(blocks):
            streams[seat].append(block)
    return [np.concatenate(blocks).astype(np.int64) for blocks in streams], decisions


def deal_bounds(game: PiquetSelfPlay, seat: int) -> list[int]:
    """Where each deal's context begins in the seat's stream, and the stream's end."""

    deals = np.asarray([meta[0] for meta in game.meta[seat]])
    return [0, *(np.flatnonzero(np.diff(deals)) + 1).tolist(), len(deals)]


class Knowledge:
    """What one seat may know, rebuilt from the events and its own private information only."""

    def __init__(self, game: PiquetSelfPlay, seat: int) -> None:
        self.game, self.seat = game, seat
        self.deal = -1

    def apply(self, event) -> None:
        seat = self.seat
        if event.kind == EV_DEAL:
            self.deal += 1
            result = self.result = self.game.engine.results[self.deal]
            self.codes = np.full(NUM_CARDS, V_UNKNOWN)
            self.codes[mask_array(result.dealt[seat])] = V_MINE
            self.shown_by_opponent = 0
            self.exchanged = 0
            return
        codes = self.codes
        mine = event.actor == seat
        if event.kind == EV_CARTE_BLANCHE and not mine:
            codes[mask_array(event.cards)] = V_OPP
            self.shown_by_opponent = event.cards
        elif event.kind == EV_SELECT and mine:
            codes[event.card] = V_SELECTED
        elif event.kind == EV_EXCHANGE:
            talon = self.result.talon
            if mine:
                codes[codes == V_SELECTED] = V_DISCARDED
                for card in talon[self.exchanged : self.exchanged + event.count]:
                    codes[card] = V_MINE
                if seat == self.result.elder:
                    for card in talon[event.count : ELDER_MAX]:
                        codes[card] = V_SEEN_TALON
            else:
                for card in cards_of(self.shown_by_opponent):
                    if codes[card] == V_OPP:
                        codes[card] = V_OPP_MAYBE
                if seat == self.result.elder:  # Younger drew the cards Elder left first.
                    for card in talon[self.exchanged : self.exchanged + event.count]:
                        if codes[card] == V_SEEN_TALON:
                            codes[card] = V_OPP
            self.exchanged += event.count
        elif event.kind in (EV_ANNOUNCE, EV_RESOLVE) and event.winner >= 0 and event.winner != seat:
            codes[mask_array(event.cards)] = V_OPP
        elif event.kind == EV_PLAY:
            codes[event.card] = V_TRICK_ME if mine else V_TRICK_OPP
            if event.winner >= 0:
                codes[codes == V_TRICK_ME] = V_PLAYED_ME
                codes[codes == V_TRICK_OPP] = V_PLAYED_OPP


def check_streams(game: PiquetSelfPlay, streams, decisions) -> None:
    engine = game.engine
    for seat in (0, 1):
        stream = streams[seat]
        bounds = deal_bounds(game, seat)
        assert len(bounds) == engine.deals + 1
        assert max(np.diff(bounds)) <= DEAL_TOKENS
        # A partie restarts the context exactly where each later deal begins.
        assert game.observed_starts[seat] == bounds[1:-1]
        assert all(stream[start, F_KIND] == EVENT_KINDS[EV_DEAL] for start in bounds[:-1])
        knowledge = Knowledge(game, seat)
        own = iter([d for d in decisions if d[0] == seat])
        events = [event for event in engine.events]
        at = 0
        upcoming = next(own, None)
        for position, token in enumerate(stream):
            if upcoming is not None and upcoming[2] == position:
                _, head, _, legal, _, seen = upcoming
                # Selections have no token; the decision token shows them.
                while at < seen:
                    assert events[at].kind == EV_SELECT
                    knowledge.apply(events[at])
                    at += 1
                assert token[F_KIND] == KIND_DECISION and token[F_HEAD] == 1 + HEADS.index(head)
                expected = knowledge.codes.copy()
                legal_cards = np.flatnonzero(legal[:NUM_CARDS]) if head != "declare" else []
                assert np.all(expected[legal_cards] == V_MINE)
                expected[legal_cards] = V_MINE_LEGAL
                assert np.array_equal(token[CARD_START:], expected), (seat, position)
                upcoming = next(own, None)
                continue
            while events[at].kind == EV_SELECT:
                knowledge.apply(events[at])
                at += 1
            event = events[at]
            at += 1
            knowledge.apply(event)
            assert token[F_KIND] == EVENT_KINDS[event.kind], (seat, position)
            assert token[F_ACTOR] == (0 if event.actor < 0 else 1 + (event.actor != seat))
            if event.kind == EV_PLAY:
                assert token[F_CARD] == event.card
            assert np.array_equal(token[CARD_START:], knowledge.codes), (seat, position, event.kind)
        assert upcoming is None
        assert all(event.kind == EV_SELECT for event in events[at:])


@pytest.mark.parametrize("mode", ["deal", "rubicon"])
def test_actor_streams_hold_exactly_the_viewers_knowledge(mode):
    games = 30 if mode == "deal" else 4
    for seed in range(games):
        for policy in (HeuristicPolicy(), RandomPolicy(seed)):
            game = PiquetSelfPlay(seed, mode=mode)
            streams, decisions = drive(game, policy)
            check_streams(game, streams, decisions)
            final = [stream[-1] for stream in streams]
            result = game.engine.results[-1]
            for seat in (0, 1):
                assert final[seat][F_KIND] == EVENT_KINDS[EV_DEAL_END]
                assert final[seat][F_MY_POINTS] == min(result.points[seat], 99)
                assert final[seat][F_OPP_POINTS] == min(result.points[1 - seat], 99)


def test_carte_blanche_and_the_left_cards_reach_the_right_seat():
    from tests.piquet.test_piquet_rules import TIE_DECK, deck

    rigged = deck(
        "7H 8H 9H 10H AH 7S 8S 9S 10S AS 7D 8D",
        "JH QH KH JS QS KS JD QD KD JC QC KC",
        "9D 10D AD 7C 8C 9C 10C AC",
    )
    for cards in (rigged, TIE_DECK):
        for policy in (HeuristicPolicy(), RandomPolicy(1)):
            game = PiquetSelfPlay(0, deck=cards, dealer=1)
            check_streams(game, *drive(game, policy))
    # Younger sees Elder's whole hand from the carte blanche on.
    game = PiquetSelfPlay(0, deck=rigged, dealer=1)
    younger_view = game.prefix_tokens()[1]
    assert younger_view[-1, F_KIND] == EVENT_KINDS[EV_CARTE_BLANCHE]
    assert (younger_view[-1, CARD_START:][mask_array(game.engine.hands[0])] == V_OPP).all()


def test_training_labels():
    game = PiquetSelfPlay(7, mode="rubicon")
    streams, decisions = drive(game, HeuristicPolicy())

    class Record:
        def __init__(self, seat, head, position):
            self.seat, self.head, self.position = seat, head, position

    class Trajectory:
        pass

    trajectory = Trajectory()
    trajectory.game, trajectory.streams = game, streams
    trajectory.decisions = [Record(seat, head, position) for seat, head, position, *_ in decisions]
    trajectory.returns = game.returns()
    sequences = training_sequences(trajectory)
    assert sequences.critic is None
    engine = game.engine
    for seat, segments in sequences.actor.items():
        # One sequence per deal: the contexts the policy saw.
        stream = streams[seat]
        assert [segment.start for segment in segments] == deal_bounds(game, seat)[:-1]
        assert np.array_equal(np.concatenate([segment.tokens for segment in segments]), stream)
        labels = {name: np.concatenate([s.labels[name] for s in segments]) for name in segments[0].labels}
        length = len(stream)
        assert labels["belief"].shape == (length, NUM_CARDS)
        uncertain = UNCERTAIN[stream[:, CARD_START:]]
        assert np.array_equal(labels["belief"] != IGNORE, uncertain)
        assert set(np.unique(labels["belief"][uncertain])) <= {BELIEF_HAND, BELIEF_DISCARD, BELIEF_TALON}
        meta = np.array(game.meta[seat])
        for position in range(length):
            deal, settled, finished, played = meta[position]
            result = engine.results[deal]
            assert labels["tricks"][position] == result.tricks[seat]
            held = result.held[1 - seat]
            assert labels["opp_suits"][position].tolist() == [
                sum(held >> (8 * s + r) & 1 for r in range(8)) for s in range(4)
            ]
            later = result.plays[played:]
            opponent = [card for player, card, _, _ in later if player != seat]
            if settled == 3 and opponent:
                assert labels["opp_next"][position] == opponent[0]
            else:
                assert labels["opp_next"][position] == IGNORE
            fate = labels["fate"][position]
            for player, card, trick_number, won in result.plays:
                assert fate[card] == (2 * trick_number + won if (player, card, trick_number, won) in later else IGNORE)
            never = set(range(NUM_CARDS)) - {card for _, card, _, _ in result.plays}
            assert all(fate[card] == 24 for card in never)
            expected = [
                IGNORE if c < settled else (2 if w < 0 else int(w != seat))
                for c, w in enumerate(result.category_winners)
            ]
            assert labels["declarations"][position].tolist() == expected
            if settled == 3 and finished < 12:
                assert labels["trick"][position] == int(result.trick_winners[finished] != seat)
            else:
                assert labels["trick"][position] == IGNORE
        first = engine.results[0].points
        assert np.allclose(labels["points"][0] * 50, [first[seat], first[1 - seat]])


def tiny_game(**game) -> PiquetGame:
    settings = dict(modes=("deal", "rubicon"), games_per_mode=3)
    settings.update(game)
    model = PiquetModelConfig(
        d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, per_layer_dim=8,
        card_width=8, card_hidden=16, card_layers=2, score_dim=16,
    )
    return PiquetGame(model, PiquetGameConfig(**settings), PiquetLossConfig())


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def replay_worst(trajectories, model_of, device) -> tuple[float, set[str]]:
    """Largest gap between recorded and replayed probabilities, and the heads seen.

    Each deal of a stream is replayed as its own context (a full forward),
    as the policy saw it; ``model_of(game, seat)`` is the seat's model.
    """

    worst, heads = 0.0, set()
    with torch.no_grad():
        for trajectory in trajectories:
            game = trajectory.game
            inner = getattr(game, "inner", game)
            for seat, stream in enumerate(trajectory.streams):
                if not len(stream):
                    continue
                actor = model_of(game, seat)
                bounds = deal_bounds(inner, seat)
                hidden, tokens = [], torch.from_numpy(stream.astype(np.int64)).to(device)
                for a, b in zip(bounds[:-1], bounds[1:]):
                    hidden.append(actor.forward_hidden(tokens[None, a:b])[0])
                    actor.pop_context()
                hidden = torch.cat(hidden)
                for decision in trajectory.decisions:
                    if decision.seat != seat:
                        continue
                    heads.add(decision.head)
                    logits = actor.policy_logits(
                        hidden[decision.position][None], decision.head, None, tokens[decision.position][None]
                    ).cpu().numpy()
                    probs = masked_probabilities(logits, (decision.probs > 0)[None])[0]
                    worst = max(worst, float(np.abs(probs - decision.probs).max()))
    return worst, heads


@pytest.mark.parametrize("cohorts", [1, 2])
@pytest.mark.parametrize("device", DEVICES)
def test_continuous_rollout_matches_full_replay(device, cohorts):
    """Cached decoding (carried-over games, refilled slots, cohorts) equals a full forward of each stream."""

    game = tiny_game()
    actor = randomize(game.build_actor().eval(), seed=3, scale=0.2).to(device)
    collector = ContinuousCollector(
        actor, device=torch.device(device), precision="fp32", kv_dtype="fp32", max_length=game.max_length,
        cohorts=cohorts,
    )
    lanes = [replace(lane, in_flight=lane.quota + 1) for lane in game.lanes(1.0)]
    rng = random.Random(0)
    trajectories = collector.collect(lanes, rng, version=1)
    assert collector.in_flight > 0
    trajectories += collector.collect(lanes, rng, version=2)
    assert any(t.game.mode == "rubicon" for t in trajectories)
    worst, heads = replay_worst(trajectories, lambda game, seat: actor, device)
    assert heads == set(HEADS)
    assert worst < (1e-5 if device == "cpu" else 1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_evaluation_games_restart_each_deal(device):
    """Against a rule-based seat (MixedGame) and between two models (MatchCollector), partie decoding equals replay."""

    from cardgym.games.piquet.evaluation import MixedGame, evaluate
    from cardgym.rl.continuous import Lane
    from cardgym.rl.match import MatchCollector

    game = tiny_game()
    first = randomize(game.build_actor().eval(), seed=3, scale=0.2).to(device)
    second = randomize(game.build_actor().eval(), seed=5, scale=0.2).to(device)
    options = dict(device=torch.device(device), precision="fp32", kv_dtype="fp32", max_length=game.max_length)

    mixed = [MixedGame(PiquetSelfPlay(seed, mode="rubicon"), seed % 2, HeuristicPolicy(), seed) for seed in range(4)]
    queue = list(mixed)
    lane = Lane("rubicon", 2, len(queue), None, lambda rng: queue.pop() if queue else None)
    collector = ContinuousCollector(first, **options)
    trajectories = collector.collect([lane], random.Random(0))
    assert len(trajectories) == 4 and all(t.game.done() for t in trajectories)
    worst, heads = replay_worst(trajectories, lambda game, seat: first, device)
    assert heads == set(HEADS) and worst < (1e-5 if device == "cpu" else 1e-3)

    matched = []
    for seed in range(4):
        inner = PiquetSelfPlay(seed, mode="rubicon")
        inner.seat_models = (seed % 2, 1 - seed % 2)
        matched.append(inner)
    lane = Lane("rubicon", 2, len(matched), None, lambda rng: matched.pop() if matched else None)
    models = [first, second]
    trajectories = MatchCollector(models, **options).collect([lane], random.Random(0))
    worst, heads = replay_worst(trajectories, lambda game, seat: models[game.seat_models[seat]], device)
    assert heads == set(HEADS) and worst < (1e-5 if device == "cpu" else 1e-3)

    against = evaluate(first, opponent="heuristic", deals=2, seed=0, device=torch.device(device), mode="rubicon",
                       against=second)
    assert against["games"] == 4 and "points_margin" in against and 0 <= against["opponent_rubiconed"] <= 1


@pytest.mark.parametrize("mode", ["deal", "rubicon"])
def test_partie_fields_carry_the_running_score(mode):
    game = PiquetSelfPlay(11, mode=mode)
    streams, _ = drive(game, HeuristicPolicy())
    engine = game.engine
    for seat, stream in enumerate(streams):
        if mode == "deal":
            assert not stream[:, list(PARTIE_FIELDS)].any()
            continue
        assert (stream[:, F_MODE] == 1).all()
        bounds = deal_bounds(game, seat)
        before = np.zeros(2, dtype=np.int64)
        for deal, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            result = engine.results[deal]
            points = np.array([result.points[seat], result.points[1 - seat]])
            assert (stream[a:b, F_DEAL] == deal).all()
            # The deal opens on the earlier deals' totals and ends on them plus its own points.
            for row, (mine, theirs) in ((a, before), (b - 1, before + points)):
                assert stream[row, F_MY_TOTAL] == total_code(mine)
                assert stream[row, F_OPP_TOTAL] == total_code(theirs)
                assert stream[row, F_LEAD] == lead_code(mine - theirs)
            assert (np.diff(stream[a:b, F_MY_TOTAL]) >= 0).all()
            before += points
        assert before.tolist() == [engine.totals[seat], engine.totals[1 - seat]]


def test_score_codes():
    totals = [total_code(t) for t in range(400)]
    assert totals[:120] == list(range(120)) and totals[-1] == TOTAL_VOCAB - 1
    assert all(a <= b for a, b in zip(totals, totals[1:]))
    leads = [lead_code(lead) for lead in range(-300, 301)]
    assert leads[0] == 0 and leads[-1] == LEAD_VOCAB - 1 and leads[300] == LEAD_VOCAB // 2
    assert all(a <= b for a, b in zip(leads, leads[1:]))
    assert [lead_code(lead) - lead_code(0) for lead in (-60, 60, 61, 70, 71)] == [-60, 60, 61, 61, 62]


def test_checkpoint_from_before_the_partie_fields_loads_blind_to_them():
    from cardgym.games.piquet.tokens import FIELD_VOCAB, LEGACY_FIELD_VOCAB

    game = tiny_game()
    source = randomize(game.build_actor().eval(), seed=2, scale=0.2)
    state = source.state_dict()
    generator = torch.Generator().manual_seed(0)
    # A legacy table: the same rows for every other field, random ones for the old partie fields.
    for name in ("fields", "ple_fields", "value_fields"):
        key = f"embedding.{name}.weight"
        if key not in state:
            continue
        table, legacy, old_at, new_at = state[key], [], 0, 0
        for field, (before, after) in enumerate(zip(LEGACY_FIELD_VOCAB, FIELD_VOCAB)):
            if field in PARTIE_FIELDS:
                legacy.append(torch.randn(before, table.shape[1], generator=generator))
            else:
                legacy.append(table[new_at : new_at + after])
            new_at += after
        state[key] = torch.cat(legacy)
        assert state[key].shape[0] == sum(LEGACY_FIELD_VOCAB)
    loaded = game.build_actor().eval()
    loaded.load_state_dict(state)
    stream = torch.from_numpy(drive(PiquetSelfPlay(3, mode="rubicon"), HeuristicPolicy())[0][0][:DEAL_TOKENS])
    blind = stream.clone()
    blind[:, list(PARTIE_FIELDS)] = 0
    with torch.no_grad():
        partie = loaded.forward_hidden(stream[None])
        single = loaded.forward_hidden(blind[None])
    assert torch.allclose(partie, single, atol=1e-6)
    # Fields outside the partie ones keep their trained rows.
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "fields" not in name:
                assert torch.equal(parameter, dict(loaded.named_parameters())[name]), name


def test_card_grid_is_suit_equivariant():
    grid = randomize(tiny_game().build_actor(), seed=1, scale=0.1).embedding.grid.eval()
    codes = torch.randint(0, CARD_VOCAB, (6, NUM_CARDS))
    permutation = torch.tensor([2, 0, 3, 1])
    cards = torch.arange(NUM_CARDS)
    moved = permutation[cards // 8] * 8 + cards % 8
    relabelled = torch.empty_like(codes)
    relabelled[:, moved] = codes
    with torch.no_grad():
        original = grid.cards(codes)
        permuted = grid.cards(relabelled)
    assert torch.allclose(permuted[:, moved], original, atol=1e-5)


def test_default_model_is_sixteen_layers_and_about_nine_million_parameters():
    actor = PiquetGame(PiquetModelConfig(), PiquetGameConfig(), PiquetLossConfig()).build_actor()
    assert actor.trunk_config.n_layers == 16
    assert 8.0e6 < sum(p.numel() for p in actor.parameters()) < 9.5e6


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the fused grid needs MPS")
@pytest.mark.parametrize("layers", [1, 2])
def test_fused_card_grid_matches_pytorch(layers):
    """The zero-padded 4 x 8 variant of the shared grid kernels against the module's own graph."""

    from cardgym.games.piquet.model import metal_grid_supported

    config = PiquetModelConfig(card_width=32, card_hidden=64, card_layers=layers, card_kernel=3)
    if not metal_grid_supported(config):
        pytest.skip("no Metal 4 tensor ops here")
    game = PiquetGame(config, PiquetGameConfig(), PiquetLossConfig())
    torch.manual_seed(0)
    grid = randomize(game.build_actor(), seed=4, scale=0.05).embedding.grid.to("mps").train()
    codes = torch.randint(0, CARD_VOCAB, (64, 40, NUM_CARDS), device="mps")
    probe = torch.randn(64, 40, NUM_CARDS, 32, device="mps")

    def run(bf16: bool, fused: bool):
        grid.fused = fused
        grid.zero_grad(set_to_none=True)
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
            cards, summary = grid(codes)
        ((cards.float() * probe).sum() + summary.float().square().sum()).backward()
        grid.fused = False
        return cards.detach().float(), {n: p.grad.float().clone() for n, p in grid.named_parameters()}

    exact, exact_grads = run(False, False)
    torch_bf16, torch_grads = run(True, False)
    fused, fused_grads = run(True, True)
    assert _relative(fused, exact) < 2 * _relative(torch_bf16, exact) + 1e-3
    assert set(fused_grads) == set(exact_grads)
    for name, grad in exact_grads.items():
        assert _relative(fused_grads[name], grad) < max(0.03, 2 * _relative(torch_grads[name], grad)), name
