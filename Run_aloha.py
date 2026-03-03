from env_aloha import ENVIRONMENT_ALOHA
from Brain import DQN
import numpy as np
import time
import os

def main(max_iter, reward_dir="rewards"):
    start = time.time()

    os.makedirs(reward_dir, exist_ok=True)

    agent_reward_list = []
    aloha_reward_list = []
    reward_list = []
    agent_state_list = []
    agent_action_list = []

    state = env.reset()
    counter2 = 0
    learn_started = False

    for i in range(max_iter):
        agent_action = dqn_agent.choose_action(state)
        observation_, reward, agent_reward, aloha_reward = env.step(agent_action)

        # build next_state with delayed observation, consistent with your original Run.py
        if i < env.DRL_delay * 2 - 1:
            next_state = env.reset()
        else:
            # state: shift window, append [previous_action, delayed_observation]
            next_state = np.concatenate([agent_state_list[counter2][2:], [agent_action_list[counter2], observation_]])
            counter2 += 1

        if i >= env.DRL_delay * 2 - 1:
            dqn_agent.store_transition(state, agent_action, reward, next_state)

        agent_state_list.append(state)
        agent_action_list.append(agent_action)

        agent_reward_list.append(agent_reward)
        aloha_reward_list.append(aloha_reward)
        reward_list.append(reward)

        # training (same pattern as your original code)
        if i > 200:
            dqn_agent.learn()
            learn_started = True

        state = next_state

        if i % 1000 == 0:
            print("iter:", i)

    # save rewards
    agent_file = os.path.join(reward_dir, "agent_aloha.txt")
    aloha_file = os.path.join(reward_dir, "aloha.txt")
    with open(agent_file, "w") as f:
        f.write(" ".join(map(str, agent_reward_list)))
    with open(aloha_file, "w") as f:
        f.write(" ".join(map(str, aloha_reward_list)))

    # summary (last 2000 slots)
    tail = 2000 if len(agent_reward_list) >= 2000 else len(agent_reward_list)
    print('-----------------------------')
    print('average agent reward: {}'.format(np.mean(agent_reward_list[-tail:])))
    print('average aloha reward: {}'.format(np.mean(aloha_reward_list[-tail:])))
    print('average total reward: {}'.format(np.mean(agent_reward_list[-tail:]) + np.mean(aloha_reward_list[-tail:])))
    print('Time elapsed:', time.time() - start)
    if not learn_started:
        print("Warning: training never started (max_iter too small).")

if __name__ == "__main__":
    # Example parameters you can tune:
    # - n_aloha_nodes: number of ALOHA stations
    # - aloha_tx_prob: offered load per station per slot
    env = ENVIRONMENT_ALOHA(
        state_size=60,
        n_aloha_nodes=1,
        aloha_tx_prob=0.9,
        seed=42,
        history_len=200
    )

    dqn_agent = DQN(
        env.state_size,
        env.n_actions,
        env.n_nodes,
        state_action_memory_size=2 * (env.DRL_delay),
        memory_size=500,
        replace_target_iter=200,
        batch_size=32,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.001,
        epsilon_decay=0.996,
    )

    main(max_iter=20000)
