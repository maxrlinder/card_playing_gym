import unittest

from cardgym.games.plump.cards import Card, Rank, Suit, make_deck
from cardgym.games.plump.rules import (
    bidding_order,
    compute_voids,
    determine_first_leader_from_bids,
    determine_trick_winner,
    legal_bids,
    legal_cards,
    score_round,
)
from cardgym.games.plump.state import Bid, ScoringConfig, Trick, TrickPlay


class RulesTest(unittest.TestCase):
    def test_deck_has_52_unique_cards(self):
        deck = make_deck()

        self.assertEqual(len(deck), 52)
        self.assertEqual(len(set(deck)), 52)

    def test_bidding_order_wraps_clockwise(self):
        self.assertEqual(bidding_order(start_player=2, num_players=4), [2, 3, 0, 1])

    def test_legal_bids_last_bidder_sum_restriction(self):
        bids = [Bid(0, 1, 0), Bid(1, 1, 1), Bid(2, 1, 2)]

        values = legal_bids(
            hand_size=4,
            existing_bids=bids,
            num_players=4,
            forbid_total_bid_equals_hand_size=True,
        )

        self.assertNotIn(1, values)
        self.assertEqual(values, [0, 2, 3, 4])

    def test_first_leader_after_bidding_breaks_tie_by_order(self):
        order = [2, 3, 0, 1]
        bids = [Bid(2, 2, 0), Bid(3, 3, 1), Bid(0, 3, 2), Bid(1, 1, 3)]

        self.assertEqual(determine_first_leader_from_bids(bids, order), 3)

    def test_follow_suit_limits_legal_cards(self):
        trick = Trick(trick_index=0, leader=0, led_suit=Suit.HEARTS)
        trick.plays.append(TrickPlay(0, Card(Suit.HEARTS, Rank.TEN), 0))
        hand = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.CLUBS, Rank.ACE)]

        self.assertEqual(legal_cards(hand, trick), [Card(Suit.HEARTS, Rank.TWO)])

    def test_cannot_follow_suit_all_cards_are_legal(self):
        trick = Trick(trick_index=0, leader=0, led_suit=Suit.HEARTS)
        trick.plays.append(TrickPlay(0, Card(Suit.HEARTS, Rank.TEN), 0))
        hand = [Card(Suit.SPADES, Rank.TWO), Card(Suit.CLUBS, Rank.ACE)]

        self.assertEqual(legal_cards(hand, trick), [Card(Suit.SPADES, Rank.TWO), Card(Suit.CLUBS, Rank.ACE)])

    def test_trick_winner_without_trump_ignores_off_suit_ace(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[
                TrickPlay(0, Card(Suit.SPADES, Rank.TWO), 0),
                TrickPlay(1, Card(Suit.SPADES, Rank.KING), 1),
                TrickPlay(2, Card(Suit.DIAMONDS, Rank.ACE), 2),
                TrickPlay(3, Card(Suit.SPADES, Rank.FIVE), 3),
            ],
        )

        self.assertEqual(determine_trick_winner(trick, trump_suit=None), 1)

    def test_trick_winner_with_trump(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[
                TrickPlay(0, Card(Suit.SPADES, Rank.KING), 0),
                TrickPlay(1, Card(Suit.DIAMONDS, Rank.TWO), 1),
                TrickPlay(2, Card(Suit.SPADES, Rank.ACE), 2),
                TrickPlay(3, Card(Suit.CLUBS, Rank.THREE), 3),
            ],
        )

        self.assertEqual(determine_trick_winner(trick, trump_suit=Suit.DIAMONDS), 1)

    def test_highest_trump_wins(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[
                TrickPlay(0, Card(Suit.SPADES, Rank.KING), 0),
                TrickPlay(1, Card(Suit.DIAMONDS, Rank.TWO), 1),
                TrickPlay(2, Card(Suit.SPADES, Rank.ACE), 2),
                TrickPlay(3, Card(Suit.DIAMONDS, Rank.JACK), 3),
            ],
        )

        self.assertEqual(determine_trick_winner(trick, trump_suit=Suit.DIAMONDS), 3)

    def test_void_inference_from_off_suit_play(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[
                TrickPlay(0, Card(Suit.SPADES, Rank.KING), 0),
                TrickPlay(1, Card(Suit.SPADES, Rank.TWO), 1),
                TrickPlay(2, Card(Suit.DIAMONDS, Rank.ACE), 2),
            ],
        )

        voids = compute_voids([trick], num_players=4)

        self.assertTrue(voids[2][Suit.SPADES])
        self.assertFalse(voids[1][Suit.SPADES])

    def test_round_scoring(self):
        scores = score_round(
            bids={0: 3, 1: 0, 2: 2},
            tricks_won={0: 3, 1: 0, 2: 1},
            scoring=ScoringConfig(hit_base_points=10, zero_bid_success_points=5, miss_points=0),
        )

        self.assertEqual(scores, {0: 13, 1: 5, 2: 0})


if __name__ == "__main__":
    unittest.main()
