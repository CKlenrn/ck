import numpy as np

DRL_delay = 5
TDMA_delay = 8
ALOHA_delay = 8
tdma_delay_tag = TDMA_delay - DRL_delay
aloha_delay_tag = ALOHA_delay - DRL_delay


class ENVIRONMENT_COEXIST(object):
    """
    Coexistence environment: DRL MAC + TDMA + ALOHA.
    One slot can have at most one successful transmission (throughput=1).
    """

    def __init__(
        self,
        state_size=60,
        n_aloha_nodes=1,
        aloha_tx_prob=0.5,
        seed=None,
        history_len=200,
    ):
        self.state_size = state_size
        self.action_space = ["w", "t"]  # wait / transmit
        self.n_actions = len(self.action_space)
        self.n_nodes = 2 + int(n_aloha_nodes)  # DRL + TDMA + ALOHA nodes

        self.DRL_delay = DRL_delay
        self.TDMA_delay = TDMA_delay
        self.ALOHA_delay = ALOHA_delay
        self.tdma_delay_tag = tdma_delay_tag
        self.aloha_delay_tag = aloha_delay_tag

        self.n_aloha_nodes = int(n_aloha_nodes)
        self.aloha_tx_prob = float(aloha_tx_prob)
        self.history_len = int(history_len)
        self.rng = np.random.RandomState(seed)

        # Ensure history buffer is long enough to cover delay tags
        min_hist = max(self.tdma_delay_tag, self.aloha_delay_tag) + 5
        if self.history_len < min_hist:
            raise ValueError(f"history_len ({self.history_len}) too small; need >= {min_hist} to cover delay tags")

        # keep TDMA pattern same as env.py
        self.tdma_action_list = [0, 0, 0, 0, 1, 1, 0, 1, 1, 1]
        self._tdma_counter = 0
        self._aloha_action_hist = np.zeros((self.history_len, self.n_aloha_nodes), dtype=np.int32)
        self._slot_counter = 0
        self._t = 0

        self.agent_reward_list = []
        self.tdma_reward_list = []
        self.aloha_reward_list = []
        self.reward_list = []
        self.DRL_observation_list = []
        self.observation_reward_counter = 0

    def reset(self):
        """Reset environment internal state for a new episode."""
        self._tdma_counter = 0
        self._slot_counter = 0
        self._aloha_action_hist.fill(0)
        self.agent_reward_list = []
        self.tdma_reward_list = []
        self.aloha_reward_list = []
        self.reward_list = []
        self.DRL_observation_list = []
        self.observation_reward_counter = 0
        return np.zeros(self.state_size, dtype=np.int32)

    def _current_tdma_action(self, t):
        """TDMA transmission that *arrives at AP* in the current DRL-aligned slot.

        At DRL send slot t, a DRL packet arrives at AP at t+DRL_delay.
        TDMA would collide in the same arrival slot if it transmitted at send slot:
            t - (TDMA_delay - DRL_delay) = t - tdma_delay_tag
        For t < tdma_delay_tag, there is no corresponding past TDMA send, so treat as 0.
        """
        if t < self.tdma_delay_tag:
            return 0
        idx = (t - self.tdma_delay_tag) % len(self.tdma_action_list)
        return int(self.tdma_action_list[idx])

    def _sample_aloha(self):
        u = self.rng.rand(self.n_aloha_nodes)
        return (u < self.aloha_tx_prob).astype(np.int32)

    def step(self, action):
        action = int(action)
        tdma_action = self._current_tdma_action(self._t)

        aloha_now = self._sample_aloha()
        self._aloha_action_hist[self._slot_counter, :] = aloha_now
        if self._t < self.aloha_delay_tag:
            aloha_effective = np.zeros(self.n_aloha_nodes, dtype=np.int32)
        else:
            aloha_idx = (self._slot_counter - self.aloha_delay_tag) % self.history_len
            aloha_effective = self._aloha_action_hist[aloha_idx, :]
        aloha_tx_count = int(np.sum(aloha_effective))

        total_tx = action + tdma_action + aloha_tx_count
        success = (total_tx == 1)

        reward = 1 if success else 0
        agent_reward = 1 if (action == 1 and success) else 0
        tdma_reward = 1 if (action == 0 and tdma_action == 1 and aloha_tx_count == 0 and success) else 0
        aloha_reward = 1 if (action == 0 and tdma_action == 0 and aloha_tx_count == 1 and success) else 0

        if action == 1:
            observation_ = 1 if success else -1
        else:
            if tdma_action + aloha_tx_count == 1:
                observation_ = 2
            else:
                observation_ = -2

        self._slot_counter = (self._slot_counter + 1) % self.history_len
        self._t += 1

        self.agent_reward_list.append(agent_reward)
        self.tdma_reward_list.append(tdma_reward)
        self.aloha_reward_list.append(aloha_reward)
        self.reward_list.append(reward)
        self.DRL_observation_list.append(observation_)
        self.observation_reward_counter += 1

        if self.observation_reward_counter >= 2 * self.DRL_delay:
            idx = self.observation_reward_counter - (2 * self.DRL_delay)
            return (
                self.DRL_observation_list[idx],
                self.reward_list[idx],
                self.agent_reward_list[idx],
                self.tdma_reward_list[idx],
                self.aloha_reward_list[idx],
            )
        return 0, 0, 0, 0, 0
