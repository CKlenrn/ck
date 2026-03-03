import os
import numpy as np
import matplotlib.pyplot as plt

import env                      # 关键：用模块变量控制 TDMA_delay
from env import ENVIRONMENT
from Brain import DQN

# ====== 你可按需改这几个参数 ======
TDMA_DELAYS = [2, 4, 6, 8]
DRL_DELAY_FIXED = 5            # 你 env.py 里就是 5  :contentReference[oaicite:8]{index=8}
TRAIN_SLOTS = 10000
TAIL = 2000                    # 跟 Run.py 一样用最后 2000 平均  :contentReference[oaicite:9]{index=9}
# =================================

def run_one_delay(tdma_delay: int) -> float:
    # 1) 修改 env.py 的全局延迟参数（ENVIRONMENT.__init__ 会读取这些全局变量）
    env.TDMA_delay = int(tdma_delay)
    env.DRL_delay = DRL_DELAY_FIXED
    env.delay_tag = env.TDMA_delay - env.DRL_delay  # :contentReference[oaicite:10]{index=10}
    env.TDMA_counter = 0                            # 重置计数器（env.py 里是 global） :contentReference[oaicite:11]{index=11}

    # 2) 创建环境
    e = ENVIRONMENT(state_size=60)

    # 3) 创建 DQN
    dqn = DQN(
        e.state_size,
        e.n_actions,
        e.n_nodes,
        state_action_memory_size=2 * e.DRL_delay,
        memory_size=500,
        replace_target_iter=200,
        batch_size=32,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.001,
        epsilon_decay=0.996,
    )

    # 4) 训练（基本照抄 Run.py 的状态拼接与延迟处理） :contentReference[oaicite:12]{index=12}
    state = e.reset()
    agent_state_list, agent_action_list = [], []
    counter2 = 0
    agent_reward_list, tdma_reward_list = [], []

    for i in range(TRAIN_SLOTS):
        a = dqn.choose_action(state)
        obs, r_total, r_agent, r_tdma = e.step(a)

        if i < e.DRL_delay * 2 - 1:
            next_state = e.reset()
        else:
            next_state = np.concatenate([
                agent_state_list[counter2][2:],
                [agent_action_list[counter2], obs]
            ])
            counter2 += 1

        if i >= e.DRL_delay * 2 - 1:
            dqn.store_transition(state, a, r_total, next_state)

        agent_state_list.append(state)
        agent_action_list.append(a)
        agent_reward_list.append(r_agent)
        tdma_reward_list.append(r_tdma)

        if i > 200:
            dqn.learn()

        state = next_state

    tail = min(TAIL, len(agent_reward_list))
    throughput = float(np.mean(agent_reward_list[-tail:]) + np.mean(tdma_reward_list[-tail:]))
    # if (i + 1) % 5000 == 0:
    # print(f"  TDMA_delay={tdma_delay}  train step {i + 1}/{TRAIN_SLOTS}")

    return throughput


def main():
    results = []
    for d in TDMA_DELAYS:
        th = run_one_delay(d)
        print(f"[TDMA coexist] TDMA_delay={d} -> throughput={th:.4f}")
        results.append(th)

    os.makedirs("results", exist_ok=True)
    out_txt = os.path.join("results", "tdma_delay_sweep.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for d, th in zip(TDMA_DELAYS, results):
            f.write(f"{d}\t{th}\n")

    # bar plot
    plt.figure()
    plt.bar([str(x) for x in TDMA_DELAYS], results)
    plt.xlabel("TDMA propagation delay (slots)")
    plt.ylabel("System throughput (avg success/slot)")
    plt.title(f"DRL MAC + TDMA coexist (DRL_delay={DRL_DELAY_FIXED})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
