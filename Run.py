from env import ENVIRONMENT
from Brain import DQN
import time
import numpy as np


def _build_next_state(i, observation, state_hist, action_hist, delayed_idx):
    if i < env.DRL_delay * 2 - 1:
        return env.reset(), delayed_idx
    next_state = np.concatenate(
        [state_hist[delayed_idx][2:], [action_hist[delayed_idx], observation]]
    )
    return next_state, delayed_idx + 1


def _run_train_loop(max_iter, learn_after=200):
    state = env.reset()
    state_hist = []
    action_hist = []
    delayed_idx = 0

    agent_rewards = []
    tdma_rewards = []

    for i in range(max_iter):
        action = dqn_agent.choose_action(state)
        observation, reward, agent_reward, tdma_reward = env.step(action)
        next_state, delayed_idx = _build_next_state(
            i, observation, state_hist, action_hist, delayed_idx
        )

        if i >= env.DRL_delay * 2 - 1:
            dqn_agent.store_transition(state, action, reward, next_state)
        if i > learn_after:
            dqn_agent.learn()

        state_hist.append(state)
        action_hist.append(action)
        agent_rewards.append(agent_reward)
        tdma_rewards.append(tdma_reward)
        state = next_state

        if i % 1000 == 0:
            print("iter:", i)

    return agent_rewards, tdma_rewards


def main(max_iter):
    start = time.time()
    agent_rewards, tdma_rewards = _run_train_loop(max_iter=max_iter)

    with open("rewards/agent_len5e4_M40.txt", "w") as f:
        f.write(" ".join(map(str, agent_rewards)))
    with open("rewards/TDMA_len5e4_M40.txt", "w") as f:
        f.write(" ".join(map(str, tdma_rewards)))

    tail = 2000 if len(agent_rewards) >= 2000 else len(agent_rewards)
    agent_avg = float(np.mean(agent_rewards[-tail:]))
    tdma_avg = float(np.mean(tdma_rewards[-tail:]))
    print("-----------------------------")
    print("average agent reward:", agent_avg)
    print("average TDMA reward:", tdma_avg)
    print("average total reward:", agent_avg + tdma_avg)
    print("Time elapsed:", time.time() - start)


if __name__ == "__main__":
    env = ENVIRONMENT(state_size=60)

    dqn_agent = DQN(env.state_size, # the number of DRL state
                    env.n_actions, # the number of DRL action
                    env.n_nodes, #  the number of nodes
                    state_action_memory_size=2*(env.DRL_delay), memory_size=500, replace_target_iter=200, batch_size=32, learning_rate=0.01,
                    gamma=0.95, epsilon=0.1, epsilon_min=0.001, epsilon_decay=0.996,
                    )
    main(max_iter=50000) # run 50000 time slots
