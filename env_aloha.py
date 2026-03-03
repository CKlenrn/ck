import numpy as np

# -----------------------------
# Coexistence: DRL MAC + ALOHA
# -----------------------------
# This file is adapted from the original "env.py" you provided, but replaces the TDMA node
# with one (or multiple) slotted-ALOHA node(s).

DRL_delay = 5          # DRL node one-way propagation delay (time slots)
ALOHA_delay = 8        # ALOHA node one-way propagation delay (time slots)
delay_tag = ALOHA_delay - DRL_delay  # >0 means ALOHA farther than DRL (arrives later at AP)

# internal ring-buffer length for ALOHA actions (must be > |delay_tag| and large enough)
DEFAULT_HISTORY_LEN = 200

class ENVIRONMENT_ALOHA(object):
    """
    Environment for "DRL-based MAC protocol coexisting with ALOHA protocol" (slotted ALOHA).

    - One DRL agent node chooses action: 0=wait, 1=transmit
    - n_aloha_nodes ALOHA node(s) transmit independently with probability aloha_tx_prob each slot
    - Channel model (simple):
        * If exactly one transmission arrives at AP in a slot -> success (reward=1)
        * Otherwise (0 or >=2) -> idle/collision (reward=0)
    - Observation encoding (same style as your original env.py):
        * action==1 (DRL tx):  1 success, -1 collision
        * action==0 (DRL wait): 2 others success, -2 idle or others collision
    - Delayed feedback:
        * Returns observation/reward after 2*DRL_delay slots, consistent with your original code.
    """

    def __init__(self,
                 state_size=60,
                 n_aloha_nodes=1,
                 aloha_tx_prob=0.2,
                 seed=None,
                 history_len=DEFAULT_HISTORY_LEN):
        self.state_size = state_size
        self.action_space = ['w', 't']  # wait / transmit
        self.n_actions = len(self.action_space)

        self.n_aloha_nodes = int(n_aloha_nodes)
        self.n_nodes = 1 + self.n_aloha_nodes  # DRL + ALOHA nodes

        self.DRL_delay = DRL_delay
        self.ALOHA_delay = ALOHA_delay
        self.delay_tag = delay_tag

        self.aloha_tx_prob = float(aloha_tx_prob)
        assert 0.0 <= self.aloha_tx_prob <= 1.0, "aloha_tx_prob must be in [0, 1]"

        self.rng = np.random.RandomState(seed)

        self.history_len = int(history_len)
        assert self.history_len >= 10, "history_len too small"
        # ring buffer: each entry is a vector of length n_aloha_nodes (0/1)
        self._aloha_action_hist = np.zeros((self.history_len, self.n_aloha_nodes), dtype=np.int32)
        self._counter = 0

        # logging lists (for delayed feedback)
        self.agent_reward_list = []
        self.aloha_reward_list = []
        self.reward_list = []
        self.DRL_observation_list = []
        self.observation_reward_counter = 0

    def reset(self):
        """Reset the DRL agent state to zeros (same as your original env.reset())."""
        init_state = np.zeros(self.state_size, dtype=np.int32)
        return init_state

    def _sample_aloha_actions(self):
        """Sample ALOHA actions for the current local time slot (slotted ALOHA)."""
        # Each ALOHA node transmits with prob p (assume always has a packet).
        u = self.rng.rand(self.n_aloha_nodes)
        return (u < self.aloha_tx_prob).astype(np.int32)

    def step(self, action):
        """
        One time slot step.
        Returns (observation, total_reward, agent_reward, aloha_reward) AFTER delay, else zeros.
        """
        action = int(action)
        assert action in (0, 1), "action must be 0(wait) or 1(transmit)"

        # 1) Sample and store ALOHA node(s) decision at this local time index
        aloha_now = self._sample_aloha_actions()
        self._aloha_action_hist[self._counter, :] = aloha_now

        # 2) Determine which ALOHA actions arrive at AP "aligned" with DRL's current slot
        # Using the same idea as your TDMA delay compensation: index shift by delay_tag.
        effective_index = (self._counter - self.delay_tag) % self.history_len
        aloha_effective = self._aloha_action_hist[effective_index, :]
        aloha_tx_count = int(np.sum(aloha_effective))

        # 3) Channel outcome at AP
        total_tx = action + aloha_tx_count
        success = (total_tx == 1)

        # Rewards (network throughput style):
        total_reward = 1 if success else 0
        agent_reward = 1 if (action == 1 and success) else 0
        # count "one ALOHA success" only when DRL waits and exactly one ALOHA transmits
        aloha_reward = 1 if (action == 0 and aloha_tx_count == 1 and success) else 0

        # Observation encoding (keep consistent with your original env.py)
        if action == 1:
            observation_ = 1 if success else -1          # tx: success or collision
        else:
            if aloha_tx_count == 1:
                observation_ = 2                          # wait + others success
            else:
                observation_ = -2                         # wait + idle or others collision

        # 4) Advance local counter
        self._counter += 1
        if self._counter >= self.history_len:
            self._counter = 0

        # 5) Store for delayed return (consistent with your original code)
        self.agent_reward_list.append(agent_reward)
        self.aloha_reward_list.append(aloha_reward)
        self.reward_list.append(total_reward)
        self.DRL_observation_list.append(observation_)
        self.observation_reward_counter += 1

        if self.observation_reward_counter >= 2 * self.DRL_delay:
            idx = self.observation_reward_counter - (2 * self.DRL_delay)
            return (self.DRL_observation_list[idx],
                    self.reward_list[idx],
                    self.agent_reward_list[idx],
                    self.aloha_reward_list[idx])
        else:
            return 0, 0, 0, 0


