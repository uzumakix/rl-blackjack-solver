import pytest
import sys
sys.path.insert(0, '.')

from mc_agent import MonteCarloAgent


class TestLearnedPolicy:
    def test_stands_on_twenty(self, trained_agent):
        # hard 20 against any dealer card, should stand (action=1)
        for dealer in range(1, 11):
            state = (20, dealer, False)
            action = trained_agent._greedy_action(state)
            assert action == 1, f"should stand on 20 vs dealer {dealer}"

    def test_stands_on_twentyone(self, trained_agent):
        for dealer in range(1, 11):
            state = (21, dealer, False)
            action = trained_agent._greedy_action(state)
            assert action == 1, f"should stand on 21 vs dealer {dealer}"

    def test_hits_on_hard_five(self, trained_agent):
        # hard 5 vs dealer 10, should hit (action=0)
        state = (5, 10, False)
        action = trained_agent._greedy_action(state)
        assert action == 0, "should hit on hard 5 vs dealer 10"


class TestQTable:
    def test_has_entries(self, trained_agent):
        assert len(trained_agent.Q) > 0, "Q table is empty after training"

    def test_visited_states_have_counts(self, trained_agent):
        total_visits = sum(trained_agent.N.values())
        assert total_visits > 0, "no states visited during training"


class TestGameValue:
    def test_house_edge(self, trained_agent):
        val = trained_agent.game_value_estimate(num_episodes=50_000)
        # house edge means value should be negative, but not absurdly so
        assert -1.0 < val < 0.5, f"game value {val} outside expected range"
