"""Training labels reconstructed by replaying a finished round.

Labels at stream position t describe the state after the token at t (every
prefix position carries the post-deal state). The true deal is used only to
build targets; it never enters an actor stream.

Actor (observer-relative seats):

* ``suit`` [L, opponents, 4]: does opponent seat r+1 still hold the suit?
  Opponents only: the observer's own suits are visible in its stream.
* ``rank_boundary`` [L, opponents + 1, 4, 2]: does the opponent (last row:
  any opponent) hold a card strictly below / above the observer's current
  lowest / highest card of that suit? Masked where the observer is void.
* ``trick_targets`` [max_players] final tricks per seat, observer included,
  with ``trick_masks`` [L, max_players, bids] marking the counts still
  feasible at each position.
* ``next_winner`` [L]: seat taking the next unresolved trick; labeled from
  the token carrying the last bid through the second-to-last trick win.
* ``bid_hit`` [max_players]: did the seat end on exactly its bid?

The critic (absolute seats) reuses seat 0's trick and next-winner labels and
adds each seat's future play order for the card-order head.
"""

from __future__ import annotations

import numpy as np

from .cards import Card
from .state import EventType, GameEvent
from .tokens import (
    IGNORE,
    NUM_CARDS,
    NUM_RANKS,
    NUM_SUITS,
    SLOT_REL_PLAYER,
    SLOT_TYPE,
    TOKEN_PLAY,
    Schema,
    card_id,
    round_events,
)

# Before the observer's 1st, 5th and 9th play of a ten-card round: the
# fixed stages at which belief accuracy is reported.
BELIEF_STAGES = (0, 4, 8)


def replay_labels(
    schema: Schema,
    initial_hands: dict[int, list[Card]],
    events: list[GameEvent],
    observer: int,
    num_players: int,
    hand_size: int,
) -> dict[str, np.ndarray]:
    total = schema.seq_len(num_players, hand_size)
    bid_count = schema.bid_count
    opponents = schema.opponents
    trick_masks = np.zeros((total, schema.max_players, bid_count), dtype=bool)
    suit = np.full((total, opponents, NUM_SUITS), IGNORE, dtype=np.int8)
    rank_boundary = np.full((total, opponents + 1, NUM_SUITS, 2), IGNORE, dtype=np.int8)
    next_winner = np.full(total, IGNORE, dtype=np.int64)
    played = round_events(events)
    winners = np.asarray(
        [(e.player - observer) % num_players for e in played if e.type == EventType.TRICK_WIN],
        dtype=np.int64,
    )

    holder = np.full(NUM_CARDS, num_players, dtype=np.int64)
    suit_count = np.zeros((num_players, NUM_SUITS), dtype=np.int64)
    # Holdings as a rank bitmask per (relative seat, suit): boundary questions
    # ("anyone below my lowest spade?") are a few integer ops on these.
    suit_bits = [0] * (num_players * NUM_SUITS)
    for player, hand in initial_hands.items():
        rel = (player - observer) % num_players
        for card in hand:
            index = card_id(card)
            holder[index] = rel
            suit_index, rank_index = divmod(index, NUM_RANKS)
            suit_count[rel, suit_index] += 1
            suit_bits[rel * NUM_SUITS + suit_index] |= 1 << rank_index
    tricks_won = np.zeros(num_players, dtype=np.int64)
    bids: list[int] = []
    bid_by_seat = np.full(num_players, -1, dtype=np.int64)
    completed = 0
    counts = np.arange(bid_count)
    full = (1 << NUM_RANKS) - 1
    cache_trick = np.zeros((schema.max_players, bid_count), dtype=bool)
    cache_suit = np.full((opponents, NUM_SUITS), IGNORE, dtype=np.int8)
    cache_boundary = np.full((opponents + 1, NUM_SUITS, 2), IGNORE, dtype=np.int8)
    # A PLAY moves holdings and a TRICK_WIN moves trick counts; track the two
    # caches apart so alternating events do not rebuild both every time.
    dirty = {"trick": True, "belief": True}

    def fill(position: int) -> None:
        if dirty["trick"]:
            cache_trick.fill(False)
            upper = np.minimum(tricks_won + (hand_size - completed), schema.max_hand_size)
            cache_trick[:num_players] = (counts[None, :] >= tricks_won[:, None]) & (
                counts[None, :] <= upper[:, None]
            )
            dirty["trick"] = False
        if dirty["belief"]:
            cache_suit.fill(IGNORE)
            cache_boundary.fill(IGNORE)
            cache_suit[: num_players - 1] = (suit_count[1:num_players] > 0).astype(np.int8)
            for suit_index in range(NUM_SUITS):
                own = suit_bits[suit_index]
                if not own:
                    continue
                below = (own & -own) - 1
                above = full & ~((1 << own.bit_length()) - 1)
                any_lower = any_higher = False
                for rel in range(1, num_players):
                    held = suit_bits[rel * NUM_SUITS + suit_index]
                    lower = (held & below) != 0
                    higher = (held & above) != 0
                    cache_boundary[rel - 1, suit_index] = (lower, higher)
                    any_lower |= lower
                    any_higher |= higher
                cache_boundary[opponents, suit_index] = (any_lower, any_higher)
            dirty["belief"] = False
        trick_masks[position] = cache_trick
        suit[position] = cache_suit
        rank_boundary[position] = cache_boundary
        if len(bids) == num_players and completed < hand_size:
            next_winner[position] = winners[completed]

    prefix = schema.prefix_len(hand_size)
    fill(0)
    trick_masks[1:prefix] = trick_masks[0]
    suit[1:prefix] = suit[0]
    rank_boundary[1:prefix] = rank_boundary[0]

    position = prefix - 1
    for index, event in enumerate(played):
        if index > 0 and event.type == EventType.BID:
            position += 1  # the bid's TURN token
            fill(position)
        position += 1
        if event.type == EventType.BID:
            bids.append(event.bid)
            bid_by_seat[(event.player - observer) % num_players] = event.bid
        elif event.type == EventType.PLAY:
            rel = (event.player - observer) % num_players
            index_of_card = card_id(event.card)
            if holder[index_of_card] != rel:
                raise AssertionError("Replay saw a play from a non-holder.")
            holder[index_of_card] = -1
            suit_index, rank_index = divmod(index_of_card, NUM_RANKS)
            suit_count[rel, suit_index] -= 1
            suit_bits[rel * NUM_SUITS + suit_index] &= ~(1 << rank_index)
            dirty["belief"] = True
        elif event.type == EventType.TRICK_WIN:
            tricks_won[(event.player - observer) % num_players] += 1
            completed += 1
            dirty["trick"] = True
        fill(position)
    if position + 1 != total or completed != hand_size:
        raise AssertionError("Replay expects a completed round.")

    trick_targets = np.full(schema.max_players, IGNORE, dtype=np.int64)
    trick_targets[:num_players] = tricks_won
    bid_hit = np.full(schema.max_players, IGNORE, dtype=np.int64)
    bid_hit[:num_players] = (tricks_won == bid_by_seat).astype(np.int64)
    return {
        "suit": suit,
        "rank_boundary": rank_boundary,
        "trick_targets": trick_targets,
        "trick_masks": trick_masks,
        "next_winner": next_winner,
        "bid_hit": bid_hit,
    }


def belief_stages(play_positions: list[int], hand_size: int) -> np.ndarray:
    """Readout positions before the observer's 1st/5th/9th play (ten-card rounds)."""

    stages = np.full(len(BELIEF_STAGES), -1, dtype=np.int64)
    if hand_size == 10:
        for index, cards_played in enumerate(BELIEF_STAGES):
            if cards_played < len(play_positions):
                stages[index] = play_positions[cards_played]
    return stages


def oracle_labels(
    schema: Schema,
    seat_zero: dict[str, np.ndarray],
    oracle: np.ndarray,
    events: list[GameEvent],
    num_players: int,
    hand_size: int,
) -> dict[str, np.ndarray]:
    """Critic labels on the oracle stream's absolute-seat axis.

    The oracle stream shares seat 0's public suffix; its longer card prefix
    describes the same post-deal state, so per-position labels broadcast the
    post-deal value across the prefix and copy the suffix.
    """

    length = oracle.shape[0]
    actor_prefix = schema.prefix_len(hand_size)
    oracle_prefix = actor_prefix + (num_players - 1) * hand_size

    def place(source: np.ndarray, fill) -> np.ndarray:
        target = np.full((length, *source.shape[1:]), fill, dtype=source.dtype)
        target[:oracle_prefix] = source[0]
        target[oracle_prefix:] = source[actor_prefix:]
        return target

    order = np.full((schema.max_players, schema.max_hand_size), NUM_CARDS, dtype=np.int8)
    slots = np.zeros(schema.max_players, dtype=np.int64)
    for event in events:
        if event.type == EventType.PLAY:
            order[event.player, slots[event.player]] = card_id(event.card)
            slots[event.player] += 1
    if not np.all(slots[:num_players] == hand_size):
        raise AssertionError("Card-order targets must cover every play.")
    # A readout includes its own token: on a PLAY row that card is played.
    is_play = oracle[:, SLOT_TYPE] == TOKEN_PLAY
    played = np.zeros((length, schema.max_players), dtype=np.int8)
    played[np.flatnonzero(is_play), oracle[is_play, SLOT_REL_PLAYER]] = 1
    return {
        "trick_targets": seat_zero["trick_targets"],
        "trick_masks": place(seat_zero["trick_masks"], False),
        "next_winner": place(seat_zero["next_winner"], IGNORE),
        "card_order": order,
        "cards_played": np.cumsum(played, axis=0, dtype=np.int8),
    }
