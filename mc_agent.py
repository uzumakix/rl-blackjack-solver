"""
mc_agent.py -- Monte Carlo Exploring Starts Agent
===================================================

Implements Monte Carlo ES (Exploring Starts) for learning the
optimal Blackjack policy. Each episode begins with a randomly
selected state-action pair, then follows the current greedy
policy. After each episode, action-value estimates Q(s,a) are
updated using first-visit Monte Carlo returns.

The learned policy converges to the optimal basic strategy as
the number of episodes grows.

Algorithm (Sutton & Barto 2018, Section 5.3):
    1. For each episode:
       a. Pick random start state and action (exploring starts)
       b. Follow greedy policy for remaining steps
       c. Compute returns G for each (s, a) visited
       d. Update Q(s, a) incrementally
    2. Derive policy: pi(s) = argmax_a Q(s, a)

State space:
    player_sum:     4..21   (18 values)
    dealer_showing: 1..10   (10 values)
    usable_ace:     0 or 1  (2 values)
    Total: 360 states

Action space:
    hit=0, stand=1, double=2
"""

import numpy as np

from blackjack_env import BlackjackEnv, HIT, STAND, DOUBLE, hand_value, draw_card

NUM_ACTIONS = 3


class MonteCarloAgent:
    """
    Monte Carlo Exploring Starts agent for Blackjack.

    Attributes
    ----------
    Q : dict
        Action-value function Q(s, a). Keys are (state, action).
    N : dict
        Visit counts N(s, a).
    policy : dict
        Current greedy policy. Maps state -> action.
    returns_history : list
        Tracks average return per batch for learning curve.
    """

    def __init__(self, seed=42):
        self.Q = {}
        self.N = {}
        self.policy = {}
        self.returns_history = []
        self.env = BlackjackEnv(seed=seed)
        self.rng = np.random.default_rng(seed)

    def _init_state_action(self, state, action):
        """Lazily initialize Q and N for a state-action pair."""
        key = (state, action)
        if key not in self.Q:
            self.Q[key] = 0.0
            self.N[key] = 0

    def _greedy_action(self, state):
        """Return the greedy action for a state, defaulting to stand."""
        if state in self.policy:
            return self.policy[state]

        best_a = STAND
        best_q = float("-inf")
        for a in range(NUM_ACTIONS):
            key = (state, a)
            q = self.Q.get(key, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    def _random_start_episode(self):
        """
        Generate an episode with exploring starts.

        The first state is dealt normally, but the first action
        is chosen uniformly at random. Subsequent actions follow
        the current greedy policy.
        """
        state, p_cards, d_cards = self.env.deal()
        p_total, _ = hand_value(p_cards)
        d_total, _ = hand_value(d_cards)

        # naturals: no decision to make
        if p_total == 21 or d_total == 21:
            if p_total == 21 and d_total == 21:
                return [(state, STAND, 0.0)]
            if p_total == 21:
                return [(state, STAND, 1.0)]
            # dealer natural beats player
            return [(state, STAND, -1.0)]

        # exploring start: random first action
        first_action = self.rng.integers(0, NUM_ACTIONS)
        trajectory = []
        done = False
        action = first_action
        step = 0

        while not done:
            next_state, reward, done, mult = self.env.step(
                p_cards, d_cards, action
            )
            # reward from env.step is already scaled by multiplier for doubles
            trajectory.append((state, action, reward))
            state = next_state
            step += 1
            if not done:
                action = self._greedy_action(state)

        return trajectory

    def train(self, num_episodes, log_interval=50_000):
        """
        Train for num_episodes using Monte Carlo Exploring Starts.

        Parameters
        ----------
        num_episodes : int
            Total training episodes.
        log_interval : int
            How often to record the average return for the
            learning curve plot.

        Returns
        -------
        list of (episode_number, average_return) tuples
        """
        batch_returns = []
        curve = []

        for ep in range(1, num_episodes + 1):
            trajectory = self._random_start_episode()

            # first-visit MC: compute return G from terminal reward
            # in Blackjack, episodes are short (few steps),
            # and only the terminal step has nonzero reward (gamma=1)
            G = sum(t[2] for t in trajectory)
            batch_returns.append(G)

            # update Q for each (s, a) visited (first-visit)
            visited = set()
            for state, action, _ in trajectory:
                sa = (state, action)
                if sa in visited:
                    continue
                visited.add(sa)
                self._init_state_action(state, action)
                self.N[sa] += 1
                # incremental mean update
                self.Q[sa] += (G - self.Q[sa]) / self.N[sa]

            # update greedy policy for visited states
            for state, _, _ in trajectory:
                best_a = STAND
                best_q = float("-inf")
                for a in range(NUM_ACTIONS):
                    q = self.Q.get((state, a), 0.0)
                    if q > best_q:
                        best_q = q
                        best_a = a
                self.policy[state] = best_a

            # log learning curve
            if ep % log_interval == 0:
                avg = np.mean(batch_returns[-log_interval:])
                curve.append((ep, avg))
                print(f"  Episode {ep:>10,d}  avg_return={avg:+.4f}")

        self.returns_history = curve
        return curve

    def get_strategy_matrix(self):
        """
        Extract the learned policy as two matrices:
        one for hard hands, one for soft hands.

        Returns
        -------
        hard : ndarray of shape (18, 10), dtype=int
            hard[i][j] = action for player_sum=i+4, dealer_showing=j+1
        soft : ndarray of shape (18, 10), dtype=int
            soft[i][j] = action for player_sum=i+4, dealer_showing=j+1,
            usable_ace=True
        """
        hard = np.full((18, 10), STAND, dtype=int)
        soft = np.full((18, 10), STAND, dtype=int)

        for p_sum in range(4, 22):
            for d_show in range(1, 11):
                i = p_sum - 4
                j = d_show - 1

                # hard hand
                state_hard = (p_sum, d_show, False)
                hard[i][j] = self._greedy_action(state_hard)

                # soft hand
                state_soft = (p_sum, d_show, True)
                soft[i][j] = self._greedy_action(state_soft)

        return hard, soft

    def game_value_estimate(self, num_episodes=500_000):
        """
        Estimate the expected return under the learned policy
        by running evaluation episodes (no exploration).
        """
        total = 0.0
        for _ in range(num_episodes):
            trajectory = self.env.simulate_episode(self._greedy_action)
            total += sum(t[2] for t in trajectory)
        return total / num_episodes
