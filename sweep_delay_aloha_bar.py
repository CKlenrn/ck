import os
import numpy as np
import matplotlib.pyplot as plt

import env_aloha                   # 关键：用模块变量控制 ALOHA_delay
from env_aloha import ENVIRONMENT_ALOHA
from Brain import DQN

# ====== 你可按需改这几个参数 ======
ALOHA_DELAYS = [2, 4, 6, 8]
DRL_DELAY_FIXED = 5           # 你当前 env_aloha.py 里就是 5  :contentReference[oaicite:5]{index=5}
P_FIXED = 0.5                 # ALOHA 发送概率（保持不变，只看 delay 影响）
TRAIN_SLOTS = 20000
EVAL_SLOTS = 5000
TAIL = 2000                   # 用最后 2000 slots 取平均（与你原 Run.py/Run_aloha.py 一致）
SEED = 42
# =================================

def run_one_delay(aloha_delay: int) -> float:
    # 1) 修改 env_aloha.py 的全局延迟参数（类初始化时会读取这些全局变量）
    env_aloha.DRL_delay = DRL_DELAY_FIXED
    env_aloha.ALOHA_delay = int(aloha_delay)
    env_aloha.delay_tag = env_aloha.ALOHA_delay - env_aloha.DRL_delay  # :contentReference[oaicite:6]{index=6}

    # 2) 创建环境
    env = ENVIRONMENT_ALOHA(
        state_size=60,
        n_aloha_nodes=1,
        aloha_tx_prob=P_FIXED,
        seed=SEED,
        history_len=200
    )

    # 3) 创建 DQN
    dqn = DQN(
        env.state_size,
        env.n_actions,
        env.n_nodes,
        state_action_memory_size=2 * env.DRL_delay,
        memory_size=500,
        replace_target_iter=200,
        batch_size=32,
        learning_rate=0.01,
        gamma=0.95,
        epsilon=0.1,
        epsilon_min=0.001,
        epsilon_decay=0.996,
    )

    # ========== TRAIN ==========
    state = env.reset()
    agent_state_list, agent_action_list = [], []
    counter2 = 0
    total_reward_train = []

    for i in range(TRAIN_SLOTS):
        a = dqn.choose_action(state)
        obs, r_total, r_agent, r_aloha = env.step(a)

        if i < env.DRL_delay * 2 - 1:
            next_state = env.reset()
        else:
            next_state = np.concatenate([
                agent_state_list[counter2][2:],
                [agent_action_list[counter2], obs]
            ])
            counter2 += 1

        if i >= env.DRL_delay * 2 - 1:
            dqn.store_transition(state, a, r_total, next_state)

        agent_state_list.append(state)
        agent_action_list.append(a)
        total_reward_train.append(r_total)

        if i > 200:
            dqn.learn()

        state = next_state

    # ========== EVAL（贪婪策略）==========
    # 注意：Brain.py 的 choose_action 每次都会 epsilon 衰减，所以 eval 需要锁死 epsilon=0
    # 这个点你另一个版本也专门提醒过 :contentReference[oaicite:7]{index=7}
    dqn.epsilon = 0.0
    dqn.epsilon_min = 0.0
    dqn.epsilon_decay = 1.0

    state = env.reset()
    agent_state_list, agent_action_list = [], []
    counter2 = 0
    total_reward_eval = []

    for i in range(EVAL_SLOTS):
        a = dqn.choose_action(state)
        obs, r_total, r_agent, r_aloha = env.step(a)

        if i < env.DRL_delay * 2 - 1:
            next_state = env.reset()
        else:
            next_state = np.concatenate([
                agent_state_list[counter2][2:],
                [agent_action_list[counter2], obs]
            ])
            counter2 += 1

        agent_state_list.append(state)
        agent_action_list.append(a)
        total_reward_eval.append(r_total)
        state = next_state

    tail = min(TAIL, len(total_reward_eval))
    throughput = float(np.mean(total_reward_eval[-tail:]))
    return throughput


def main():
    results = []
    for d in ALOHA_DELAYS:
        th = run_one_delay(d)
        print(f"[ALOHA coexist] ALOHA_delay={d} -> throughput={th:.4f}")
        results.append(th)

    os.makedirs("results", exist_ok=True)
    out_txt = os.path.join("results", "aloha_delay_sweep.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for d, th in zip(ALOHA_DELAYS, results):
            f.write(f"{d}\t{th}\n")

    # bar plot
    plt.figure()
    plt.bar([str(x) for x in ALOHA_DELAYS], results)
    plt.xlabel("ALOHA propagation delay (slots)")
    plt.ylabel("System throughput (avg success/slot)")
    plt.title(f"DRL MAC + ALOHA coexist (p={P_FIXED}, DRL_delay={DRL_DELAY_FIXED})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
