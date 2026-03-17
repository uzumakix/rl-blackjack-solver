import pytest
import sys
sys.path.insert(0, '.')
from blackjack_env import BlackjackEnv
from mc_agent import MonteCarloAgent

@pytest.fixture
def env():
    return BlackjackEnv(seed=42)

@pytest.fixture
def trained_agent():
    agent = MonteCarloAgent(seed=42)
    agent.train(num_episodes=100_000, log_interval=100_000)
    return agent
