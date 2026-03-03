import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 你只需要改这里 =====
P_FIXED = 0.9   # ALOHA 的发送概率 p（例如 0.6）
REWARD_DIR = "rewards"
MAX_SLOTS = 50000   # 论文图一般到 5e4
# =========================

def cumulative_avg(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    return np.cumsum(x) / np.arange(1, len(x) + 1, dtype=float)

def load_total_trace(dirpath):
    # 优先读 eval（新版 Run_aloha.py 输出）
    cands = [
        ("agent_eval.txt", "aloha_eval.txt"),
        ("agent_aloha.txt", "aloha.txt"),     # 旧版输出
        ("agent_train.txt", "aloha_train.txt")
    ]
    for a, b in cands:
        fa, fb = os.path.join(dirpath, a), os.path.join(dirpath, b)
        if os.path.exists(fa) and os.path.exists(fb):
            agent = np.loadtxt(fa)
            aloha = np.loadtxt(fb)
            L = min(len(agent), len(aloha))
            return agent[:L] + aloha[:L], (a, b)
    raise FileNotFoundError("找不到 rewards 文件，请确认 rewards 文件夹里有 agent/aloha 的 txt 输出。")

def main():
    total, used = load_total_trace(REWARD_DIR)
    total = total[:min(len(total), MAX_SLOTS)]

    total_avg = cumulative_avg(total)
    x = np.arange(len(total_avg))

    optimal = max(P_FIXED, 1.0 - P_FIXED)

    plt.plot(x, total_avg, linewidth=1.2, label="Total")
    plt.axhline(optimal, linestyle="--", linewidth=2.0, label="Optimal")

    plt.xlabel("Time slot")
    plt.ylabel("Throughput")
    plt.ylim((0.0, 1.05))
    plt.grid(False)
    plt.legend(loc="best")
    plt.tight_layout()

    print("Loaded traces:", used, "p =", P_FIXED, "optimal =", optimal)
    plt.show()

if __name__ == "__main__":
    main()
