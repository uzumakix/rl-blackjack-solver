"""
blackjack_env.py -- Blackjack MDP Environment
===============================================

Simulates standard casino Blackjack with the following rules:
    - Infinite deck (cards drawn with replacement)
    - Dealer stands on soft 17
    - Natural blackjack pays 1.0 (simplified, no 3:2)
    - Player actions: hit (0), stand (1), double (2)
    - Double: draw exactly one card, bet is doubled (first two cards only)
    - No splitting, no insurance, no surrender

State representation:
    (player_sum, dealer_showing, usable_ace)
    player_sum:      4..21
    dealer_showing:  1..10 (1 = Ace)
    usable_ace:      bool (True if player holds a usable ace)

Reference: Sutton & Barto (2018), Example 5.1 / 5.3.
"""

import numpy as np

# card values: 1=Ace, 2..10=face value, 10=face card
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

HIT = 0
STAND = 1
DOUBLE = 2
ACTION_NAMES = {0: "H", 1: "S", 2: "D"}


def draw_card(rng):
    """Draw a single card from an infinite deck."""
    return rng.choice(CARD_VALUES)


def hand_value(cards):
    """
    Compute the best hand value and whether an ace is usable.

    Returns (total, usable_ace).
    A usable ace counts as 11 without busting.
    """
    total = sum(cards)
    usable = False
    if 1 in cards and total + 10 <= 21:
        total += 10
        usable = True
    return total, usable


def dealer_play(dealer_cards, rng):
    """
    Dealer draws until reaching 17 or higher.
    Stands on soft 17.
    """
    total, usable = hand_value(dealer_cards)
    while total < 17:
        dealer_cards.append(draw_card(rng))
        total, usable = hand_value(dealer_cards)
    return total


class BlackjackEnv:
    """
    Blackjack environment for episodic RL.

    Each episode starts with a random deal. The player acts until
    standing, doubling, or busting. Then the dealer plays by fixed
    rules. Returns +1 (win), -1 (loss), or 0 (draw).
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def deal(self):
        """
        Deal initial hands. Returns (state, player_cards, dealer_cards).
        State = (player_sum, dealer_showing, usable_ace).
        """
        player_cards = [draw_card(self.rng), draw_card(self.rng)]
        dealer_cards = [draw_card(self.rng), draw_card(self.rng)]

        p_total, p_usable = hand_value(player_cards)
        dealer_showing = dealer_cards[0]

        state = (p_total, dealer_showing, p_usable)
        return state, player_cards, dealer_cards

    def step(self, player_cards, dealer_cards, action):
        """
        Execute one action.

        Parameters
        ----------
        player_cards : list[int]
        dealer_cards : list[int]
        action : int
            0=hit, 1=stand, 2=double

        Returns
        -------
        next_state : tuple or None (if terminal)
        reward : float (already multiplied for doubles)
        done : bool
        multiplier : float (1.0 normally, 2.0 after double)
        """
        multiplier = 1.0

        # doubling only legal on initial two-card hand; treat as hit otherwise
        if action == DOUBLE and len(player_cards) != 2:
            action = HIT

        if action == DOUBLE:
            player_cards.append(draw_card(self.rng))
            multiplier = 2.0
            p_total, p_usable = hand_value(player_cards)
            if p_total > 21:
                return None, -1.0 * multiplier, True, multiplier
            # forced stand after double
            d_total = dealer_play(dealer_cards, self.rng)
            reward = self._compare(p_total, d_total)
            return None, reward * multiplier, True, multiplier

        if action == HIT:
            player_cards.append(draw_card(self.rng))
            p_total, p_usable = hand_value(player_cards)
            if p_total > 21:
                return None, -1.0, True, multiplier
            state = (p_total, dealer_cards[0], p_usable)
            return state, 0.0, False, multiplier

        # STAND
        p_total, _ = hand_value(player_cards)
        d_total = dealer_play(dealer_cards, self.rng)
        reward = self._compare(p_total, d_total)
        return None, reward, True, multiplier

    @staticmethod
    def _compare(player_total, dealer_total):
        """Compare final hands. Returns +1, -1, or 0."""
        if dealer_total > 21:
            return 1.0
        if player_total > dealer_total:
            return 1.0
        if player_total < dealer_total:
            return -1.0
        return 0.0

    def simulate_episode(self, policy_fn):
        """
        Run a full episode following policy_fn(state) -> action.

        Returns list of (state, action, reward) tuples for the episode.
        """
        state, p_cards, d_cards = self.deal()

        # check naturals (two-card 21)
        p_total, _ = hand_value(p_cards)
        d_total, _ = hand_value(d_cards)
        if p_total == 21 or d_total == 21:
            if p_total == 21 and d_total == 21:
                return [(state, STAND, 0.0)]
            if p_total == 21:
                return [(state, STAND, 1.0)]
            # dealer natural, player loses immediately
            return [(state, STAND, -1.0)]

        trajectory = []
        done = False
        while not done:
            action = policy_fn(state)
            next_state, reward, done, mult = self.step(
                p_cards, d_cards, action
            )
            # reward from step() is already scaled by multiplier for doubles
            trajectory.append((state, action, reward))
            state = next_state

        return trajectory
