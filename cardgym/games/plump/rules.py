"""Pure rule helpers for Plump."""

from __future__ import annotations

from .cards import Card, Suit, sort_cards
from .state import Bid, ScoringConfig, Trick


def bidding_order(start_player: int, num_players: int) -> list[int]:
    """Return clockwise bidding order from an absolute start player."""

    return [(start_player + offset) % num_players for offset in range(num_players)]


def legal_bids(
    hand_size: int,
    existing_bids: list[Bid],
    num_players: int,
    forbid_total_bid_equals_hand_size: bool = True,
) -> list[int]:
    """Return legal bid values for the next bidder."""

    values = list(range(hand_size + 1))
    is_last_bidder = len(existing_bids) == num_players - 1
    if forbid_total_bid_equals_hand_size and is_last_bidder:
        forbidden = hand_size - sum(bid.value for bid in existing_bids)
        values = [value for value in values if value != forbidden]
    return values


def legal_cards(hand: list[Card], current_trick: Trick | None) -> list[Card]:
    """Return cards legal to play from ``hand`` in the current trick."""

    if current_trick is None or not current_trick.plays or current_trick.led_suit is None:
        return sort_cards(hand)
    suited_cards = [card for card in hand if card.suit == current_trick.led_suit]
    return sort_cards(suited_cards if suited_cards else hand)


def determine_trick_winner(trick: Trick, trump_suit: Suit | None) -> int:
    """Determine the winner of a complete trick without mutating it."""

    if not trick.plays:
        raise ValueError("Cannot determine winner of an empty trick.")
    if trick.led_suit is None:
        raise ValueError("Cannot determine winner without a led suit.")

    trump_plays = [play for play in trick.plays if trump_suit is not None and play.card.suit == trump_suit]
    if trump_plays:
        return max(trump_plays, key=lambda play: int(play.card.rank)).player

    led_plays = [play for play in trick.plays if play.card.suit == trick.led_suit]
    if not led_plays:
        raise ValueError("No played card matches the led suit.")
    return max(led_plays, key=lambda play: int(play.card.rank)).player


def determine_first_leader_from_bids(bids: list[Bid], order: list[int]) -> int:
    """Choose the highest bidder, breaking ties by earliest bidding order."""

    if not bids:
        raise ValueError("Cannot determine first leader without bids.")
    bid_by_player = {bid.player: bid.value for bid in bids}
    max_bid = max(bid_by_player.values())
    for player in order:
        if bid_by_player[player] == max_bid:
            return player
    raise ValueError("Bidding order does not contain all bidders.")


def compute_voids(tricks: list[Trick], num_players: int) -> dict[int, dict[Suit, bool]]:
    """Infer public void information from completed and current trick plays."""

    voids = {player: {suit: False for suit in Suit} for player in range(num_players)}
    for trick in tricks:
        if trick.led_suit is None:
            continue
        for play in trick.plays:
            if play.card.suit != trick.led_suit:
                voids[play.player][trick.led_suit] = True
    return voids


def score_round(
    bids: dict[int, int],
    tricks_won: dict[int, int],
    scoring: ScoringConfig,
) -> dict[int, int]:
    """Score a round using configurable Swedish Plump scoring."""

    scores: dict[int, int] = {}
    for player, bid in bids.items():
        won = tricks_won.get(player, 0)
        if won == bid:
            scores[player] = scoring.zero_bid_success_points if bid == 0 else scoring.hit_base_points + bid
        else:
            scores[player] = scoring.miss_points
    return scores


def relative_rewards(scores: dict[int, int]) -> dict[int, float]:
    """Each player's score minus the mean opponent score (zero-sum)."""

    if len(scores) < 2:
        raise ValueError("Relative rewards require at least two players.")
    total = sum(scores.values())
    opponents = len(scores) - 1
    return {player: float(score - (total - score) / opponents) for player, score in scores.items()}
