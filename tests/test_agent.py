import pytest
import numpy as np
import sys
sys.path.insert(0, '.')

from mc_agent import MonteCarloAgent
from blackjack_env import BlackjackEnv


class TestLearnedPolicy:
    def test_stands_on_twenty(self, trained_agent):
        # hard 20 against any dealer card, should always stand
        for dealer in range(1, 11):
            state = (20, dealer, False)
            action = trained_agent.best_action(state)
            assert action == 0, f"should stand on 20 vs dealer {dealer}"

    def test_stands_on_twentyone(self, trained_agent):
        for dealer in range(1, 11):
            state = (21, dealer, False)
            action = trained_agent.best_action(state)
            assert action == 0, f"should stand on 21 vs dealer {dealer}"

    def test_hits_on_hard_five(self, trained_agent):
        # hard 5 vs dealer 10, always hit
        state = (5, 10, False)
        action = trained_agent.best_action(state)
        assert action == 1, "should hit on hard 5 vs dealer 10"


class TestQTable:
    def test_shape(self, trained_agent):
        # Q(player_sum, dealer_card, usable_ace, action)
        # player: 4-21 (18), dealer: 1-10 (10), usable: 0/1 (2), action: 0/1/2 or 0/1
        shape = trained_agent.Q.shape
        assert shape[0] == 18, f"player dim {shape[0]} expected 18"
        assert shape[1] == 10, f"dealer dim {shape[1]} expected 10"
        assert shape[2] == 2, f"usable ace dim {shape[2]} expected 2"

    def test_visited_states_have_counts(self, trained_agent):
        # at least some states should have been visited
        total_visits = trained_agent.N.sum()
        assert total_visits > 0, "no states visited during training"


class TestGameValue:
    def test_house_edge(self, trained_agent):
        env = BlackjackEnv()
        val = trained_agent.game_value_estimate(env, num_episodes=10_000)
        # house edge means value should be negative, but not worse than -1
        assert -1.0 < val < 0.0, f"game value {val} outside expected range"
