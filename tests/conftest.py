import pytest
import sys
sys.path.insert(0, '.')
from blackjack_env import BlackjackEnv
from mc_agent import MonteCarloAgent

@pytest.fixture
def env():
    return BlackjackEnv()

@pytest.fixture
def trained_agent():
    env = BlackjackEnv()
    agent = MonteCarloAgent()
    agent.train(env, num_episodes=100_000, log_interval=100_000)
    return agent
