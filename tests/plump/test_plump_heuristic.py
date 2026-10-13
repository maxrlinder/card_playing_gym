import unittest

from cardgym.games.plump.cards import Card, Rank, Suit
from cardgym.games.plump.heuristic import (
    _adjust_distribution_for_prior_bids,
    _lost_player_should_take,
    _rank_only_trick_distribution,
    _select_card_for_intent,
    _select_expected_score_bid,
    _select_heuristic_play_values,
)
from cardgym.games.plump.state import Bid, Trick, TrickPlay


def _mean(distribution: dict[int, float]) -> float:
    return sum(tricks * probability for tricks, probability in distribution.items())


class HeuristicBiddingTest(unittest.TestCase):
    def test_more_opponent_cards_reduce_rank_only_strength(self):
        hand = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.THREE),
        ]

        means = [
            _mean(_rank_only_trick_distribution(hand, num_players=num_players))
            for num_players in (3, 4, 5)
        ]

        self.assertGreater(means[0], means[1])
        self.assertGreater(means[1], means[2])

    def test_prior_bids_exploit_weak_signals_and_respect_strong_signals(self):
        hand = [
            Card(Suit.SPADES, Rank.QUEEN),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        base = _rank_only_trick_distribution(hand, num_players=5)
        low = _adjust_distribution_for_prior_bids(
            base,
            prior_bids=[0, 0, 0],
            hand_size=3,
            num_players=5,
            strength=0.12,
            max_signal=2.0,
        )
        high = _adjust_distribution_for_prior_bids(
            base,
            prior_bids=[1, 1, 1],
            hand_size=3,
            num_players=5,
            strength=0.12,
            max_signal=2.0,
        )

        self.assertGreater(_mean(low), _mean(base))
        self.assertGreater(_mean(base), _mean(high))

    def test_bid_signal_cannot_remove_guaranteed_top_run(self):
        hand = [
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.SPADES, Rank.KING),
            Card(Suit.SPADES, Rank.QUEEN),
        ]
        base = _rank_only_trick_distribution(hand, num_players=5)
        adjusted = _adjust_distribution_for_prior_bids(
            base,
            prior_bids=[3, 3, 3, 3],
            hand_size=3,
            num_players=5,
            strength=0.12,
            max_signal=2.0,
        )

        self.assertEqual(base, {3: 1.0})
        self.assertEqual(adjusted, {3: 1.0})
        self.assertEqual(_select_expected_score_bid(adjusted, [0, 1, 2, 3]), 3)

    def test_expected_score_values_one_trick_more_than_zero(self):
        distribution = {0: 0.55, 1: 0.45}

        self.assertEqual(_select_expected_score_bid(distribution, [0, 1]), 1)

    def test_illegal_best_bid_falls_back_to_best_legal_score(self):
        distribution = {0: 0.1, 1: 0.7, 2: 0.2}

        self.assertEqual(_select_expected_score_bid(distribution, [0, 2]), 2)


class HeuristicPlayTest(unittest.TestCase):
    def test_take_mode_plays_lowest_when_no_card_can_win_current_trick(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[TrickPlay(0, Card(Suit.SPADES, Rank.ACE), 0)],
        )

        card = _select_card_for_intent(
            [Card(Suit.SPADES, Rank.TWO), Card(Suit.SPADES, Rank.KING)],
            player=1,
            current_trick=trick,
            trump_suit=None,
            wants_trick=True,
        )

        self.assertEqual(card, Card(Suit.SPADES, Rank.TWO))

    def test_take_mode_plays_highest_current_winner(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[TrickPlay(0, Card(Suit.SPADES, Rank.QUEEN), 0)],
        )

        card = _select_card_for_intent(
            [Card(Suit.SPADES, Rank.KING), Card(Suit.SPADES, Rank.ACE)],
            player=1,
            current_trick=trick,
            trump_suit=None,
            wants_trick=True,
        )

        self.assertEqual(card, Card(Suit.SPADES, Rank.ACE))

    def test_avoid_mode_plays_highest_non_winner(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[TrickPlay(0, Card(Suit.SPADES, Rank.ACE), 0)],
        )

        card = _select_card_for_intent(
            [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.KING)],
            player=1,
            current_trick=trick,
            trump_suit=None,
            wants_trick=False,
        )

        self.assertEqual(card, Card(Suit.SPADES, Rank.KING))

    def test_avoid_mode_plays_lowest_when_all_cards_win(self):
        trick = Trick(
            trick_index=0,
            leader=0,
            led_suit=Suit.SPADES,
            plays=[TrickPlay(0, Card(Suit.SPADES, Rank.FIVE), 0)],
        )

        card = _select_card_for_intent(
            [Card(Suit.SPADES, Rank.TEN), Card(Suit.SPADES, Rank.KING)],
            player=1,
            current_trick=trick,
            trump_suit=None,
            wants_trick=False,
        )

        self.assertEqual(card, Card(Suit.SPADES, Rank.TEN))

    def test_lost_player_offloads_when_current_winner_is_on_bid(self):
        self.assertFalse(
            _lost_player_should_take(
                player=0,
                bids={0: 0, 1: 1, 2: 0},
                tricks_won={0: 1, 1: 1, 2: 0},
                remaining_tricks=2,
                current_winner=1,
                num_players=3,
                total_round_tricks=3,
            )
        )

    def test_lost_player_steals_when_current_winner_still_needs_trick(self):
        self.assertTrue(
            _lost_player_should_take(
                player=0,
                bids={0: 0, 1: 1, 2: 0},
                tricks_won={0: 1, 1: 0, 2: 0},
                remaining_tricks=2,
                current_winner=1,
                num_players=3,
                total_round_tricks=3,
            )
        )

    def test_overbid_gone_over_uses_take_mode(self):
        observation = _play_observation(
            hand_size=3,
            bids=[Bid(0, 0, 0), Bid(1, 2, 1), Bid(2, 2, 2)],
            tricks_won={0: 1, 1: 0, 2: 0},
            legal_cards=[Card(Suit.SPADES, Rank.TWO), Card(Suit.SPADES, Rank.KING)],
            current_trick=Trick(
                trick_index=1,
                leader=1,
                led_suit=Suit.SPADES,
                plays=[TrickPlay(1, Card(Suit.SPADES, Rank.QUEEN), 0)],
            ),
        )

        self.assertEqual(
            observation,
            Card(Suit.SPADES, Rank.KING),
        )

    def test_underbid_unreachable_uses_avoid_mode(self):
        observation = _play_observation(
            hand_size=4,
            bids=[Bid(0, 3, 0), Bid(1, 0, 1), Bid(2, 0, 2)],
            tricks_won={0: 0, 1: 0, 2: 0},
            legal_cards=[Card(Suit.SPADES, Rank.TWO), Card(Suit.SPADES, Rank.KING)],
            current_trick=Trick(
                trick_index=2,
                leader=1,
                led_suit=Suit.SPADES,
                plays=[TrickPlay(1, Card(Suit.SPADES, Rank.QUEEN), 0)],
            ),
        )

        self.assertEqual(
            observation,
            Card(Suit.SPADES, Rank.TWO),
        )


def _play_observation(
    *,
    hand_size: int,
    bids: list[Bid],
    tricks_won: dict[int, int],
    legal_cards: list[Card],
    current_trick: Trick,
) -> Card:
    """The card seat 0 of a three-player round plays holding exactly ``legal_cards``."""

    return _select_heuristic_play_values(
        player=0,
        hand_size=hand_size,
        remaining_tricks=len(legal_cards),
        bids=bids,
        tricks_won=tricks_won,
        current_trick=current_trick,
        trump_suit=None,
        legal_cards=list(legal_cards),
        num_players=3,
    )


if __name__ == "__main__":
    unittest.main()
