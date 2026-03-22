import pytest
import sys
sys.path.insert(0, '.')

from blackjack_env import BlackjackEnv, draw_card, hand_value
import numpy as np


class TestDrawCard:
    def test_card_in_valid_range(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            card = draw_card(rng)
            assert 1 <= card <= 10, f"drew {card}, outside 1-10"


class TestHandValue:
    def test_blackjack_ace_ten(self):
        val, usable = hand_value([1, 10])
        assert val == 21
        assert usable is True

    def test_two_aces_and_nine(self):
        val, usable = hand_value([1, 1, 9])
        assert val == 21
        assert usable is True

    def test_bust_no_ace(self):
        val, usable = hand_value([10, 10, 5])
        assert val == 25
        assert usable is False

    def test_soft_sixteen(self):
        val, usable = hand_value([1, 5])
        assert val == 16
        assert usable is True

    def test_ace_forced_to_one(self):
        val, usable = hand_value([1, 10, 10])
        assert val == 21
        assert usable is False


class TestDeal:
    def test_returns_valid_state(self):
        env = BlackjackEnv(seed=1)
        state, p_cards, d_cards = env.deal()
        player_sum, dealer_card, usable_ace = state
        assert 2 <= player_sum <= 21
        assert 1 <= dealer_card <= 10
        assert isinstance(usable_ace, (bool, int))


class TestStep:
    def test_stand_terminates(self):
        env = BlackjackEnv(seed=1)
        state, p_cards, d_cards = env.deal()
        _, _, done, _ = env.step(p_cards, d_cards, 1)  # stand
        assert done is True

    def test_hit_gives_valid_state(self):
        env = BlackjackEnv(seed=1)
        state, p_cards, d_cards = env.deal()
        next_state, reward, done, mult = env.step(p_cards, d_cards, 0)  # hit
        if not done:
            player_sum, dealer_card, usable = next_state
            assert 2 <= player_sum <= 31
            assert 1 <= dealer_card <= 10
