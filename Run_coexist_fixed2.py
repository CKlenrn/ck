from env_coexist_fixed2 import ENVIRONMENT_COEXIST
from Brain import DQN
import numpy as np
import time
import os
import argparse


def _build_next_state(i, observation, state_hist, action_hist, delayed_idx, env, zero_state):
    """Build next state using delayed alignment (no environment reset during warm-up)."""
    if i < env.DRL_delay * 2 - 1:
        return zero_state.copy(), delayed_idx
    next_state = np.concatenate(
        [state_hist[delayed_idx][2:], [action_hist[delayed_idx], observation]]
    )
    return next_state, delayed_idx + 1


def _run_train_loop(env, dqn_agent, max_iter, learn_after=200):
    state = env.reset()
    zero_state = np.zeros(env.state_size, dtype=np.int32)
    state_hist = []
    action_hist = []
    delayed_idx = 0

    agent_rewards = []
    tdma_rewards = []
    aloha_rewards = []
    total_rewards = []

    for i in range(max_iter):
        action = dqn_agent.choose_action(state)
        observation, reward, agent_reward, tdma_reward, aloha_reward = env.step(action)
        next_state, delayed_idx = _build_next_state(
            i, observation, state_hist, action_hist, delayed_idx, env, zero_state
        )

        if i >= env.DRL_delay * 2 - 1:
            dqn_agent.store_transition(state, action, reward, next_state)
        if i > learn_after:
            dqn_agent.learn()

        state_hist.append(state)
        action_hist.append(action)
        agent_rewards.append(agent_reward)
        tdma_rewards.append(tdma_reward)
        aloha_rewards.append(aloha_reward)
        total_rewards.append(reward)
        state = next_state

        if i % 1000 == 0:
            print("iter:", i)

    return agent_rewards, tdma_rewards, aloha_rewards, total_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=30000)
    parser.add_argument("--state_size", type=int, default=60)
    parser.add_argument("--n_aloha_nodes", type=int, default=1)
    parser.add_argument("--aloha_tx_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward_dir", type=str, default="rewards")
    args = parser.parse_args()

    start = time.time()
    os.makedirs(args.reward_dir, exist_ok=True)

    env = ENVIRONMENT_COEXIST(
        state_size=args.state_size,
        n_aloha_nodes=args.n_aloha_nodes,
        aloha_tx_prob=args.aloha_tx_prob,
        seed=args.seed,
        history_len=200,
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

    agent_rewards, tdma_rewards, aloha_rewards, total_rewards = _run_train_loop(
        env=env, dqn_agent=dqn_agent, max_iter=args.max_iter
    )

    with open(f"{args.reward_dir}/agent_coexist.txt", "w") as f:
        f.write(" ".join(map(str, agent_rewards)))
    with open(f"{args.reward_dir}/tdma_coexist.txt", "w") as f:
        f.write(" ".join(map(str, tdma_rewards)))
    with open(f"{args.reward_dir}/aloha_coexist.txt", "w") as f:
        f.write(" ".join(map(str, aloha_rewards)))
    with open(f"{args.reward_dir}/total_coexist.txt", "w") as f:
        f.write(" ".join(map(str, total_rewards)))

    tail = 2000 if len(total_rewards) >= 2000 else len(total_rewards)
    print("-----------------------------")
    print("average agent throughput:", float(np.mean(agent_rewards[-tail:])))
    print("average TDMA throughput:", float(np.mean(tdma_rewards[-tail:])))
    print("average ALOHA throughput:", float(np.mean(aloha_rewards[-tail:])))
    print("average total throughput:", float(np.mean(total_rewards[-tail:])))
    print("Time elapsed:", time.time() - start)


if __name__ == "__main__":
    main()
