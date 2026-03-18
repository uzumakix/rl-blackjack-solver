import pytest
import sys
sys.path.insert(0, '.')

from blackjack_env import BlackjackEnv


class TestDrawCard:
    def test_card_in_valid_range(self):
        env = BlackjackEnv()
        for _ in range(200):
            card = env.draw_card()
            assert 1 <= card <= 10, f"drew {card}, outside 1-10"


class TestHandValue:
    def test_blackjack_ace_ten(self):
        env = BlackjackEnv()
        val, usable = env.hand_value([1, 10])
        assert val == 21
        assert usable is True

    def test_two_aces_and_nine(self):
        env = BlackjackEnv()
        val, usable = env.hand_value([1, 1, 9])
        assert val == 21
        assert usable is True

    def test_bust_no_ace(self):
        env = BlackjackEnv()
        val, usable = env.hand_value([10, 10, 5])
        assert val == 25
        assert usable is False

    def test_soft_sixteen(self):
        env = BlackjackEnv()
        val, usable = env.hand_value([1, 5])
        assert val == 16
        assert usable is True

    def test_ace_forced_to_one(self):
        env = BlackjackEnv()
        val, usable = env.hand_value([1, 10, 10])
        assert val == 21
        assert usable is False


class TestDeal:
    def test_returns_valid_state(self):
        env = BlackjackEnv()
        state = env.deal()
        assert isinstance(state, tuple)
        player_sum, dealer_card, usable_ace = state
        assert 2 <= player_sum <= 21
        assert 1 <= dealer_card <= 10
        assert isinstance(usable_ace, (bool, int))


class TestStep:
    def test_stand_terminates(self):
        env = BlackjackEnv()
        env.deal()
        # action 0=hit, 1=stand, 2=double
        _, _, done = env.step(1)  # stand
        assert done is True

    def test_hit_gives_valid_state(self):
        env = BlackjackEnv()
        env.deal()
        next_state, reward, done = env.step(0)  # hit
        if not done:
            player_sum, dealer_card, usable = next_state
            assert 2 <= player_sum <= 31
            assert 1 <= dealer_card <= 10
