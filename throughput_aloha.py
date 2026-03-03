# throughput_aloha.py  (ONLY 2 LINES: total and optimal)
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def moving_avg(x, N=2000):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    out = np.zeros_like(x, dtype=float)
    s = 0.0
    for i in range(len(x)):
        if i < N:
            s += x[i]
            out[i] = s / (i + 1)
        else:
            s += x[i] - x[i - N]
            out[i] = s / N
    return out


def load_rewards(reward_dir):
    """
    Priority:
      1) agent_eval.txt + aloha_eval.txt  (new eval files)
      2) agent_aloha.txt + aloha.txt      (legacy files)
    Returns: agent, aloha (np arrays)
    """
    cand = [
        ("agent_eval.txt", "aloha_eval.txt"),
        ("agent_aloha.txt", "aloha.txt"),
    ]
    for a, b in cand:
        fa = os.path.join(reward_dir, a)
        fb = os.path.join(reward_dir, b)
        if os.path.exists(fa) and os.path.exists(fb):
            agent = np.loadtxt(fa)
            aloha = np.loadtxt(fb)
            return agent, aloha
    raise FileNotFoundError(
        "No reward files found. Expected either:\n"
        "  agent_eval.txt + aloha_eval.txt\n"
        "or legacy:\n"
        "  agent_aloha.txt + aloha.txt\n"
        f"under directory: {reward_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="rewards", help="reward directory, e.g. rewards")
    ap.add_argument("--N", type=int, default=2000, help="moving average window length")
    ap.add_argument("--p", type=float, required=True, help="ALOHA transmit probability p (for optimal line)")
    args = ap.parse_args()

    agent, aloha = load_rewards(args.dir)

    L = min(len(agent), len(aloha))
    agent = agent[:L]
    aloha = aloha[:L]

    total = agent + aloha
    total_ma = moving_avg(total, N=args.N)

    # Theoretical optimal (single ALOHA node case from the paper): max(p, 1-p)
    p = float(args.p)
    optimal = np.ones(L) * max(p, 1.0 - p)

    total_line, = plt.plot(total_ma, lw=1.8, label="total (MA)")
    opt_line, = plt.plot(optimal, lw=2.2, label="optimal")

    plt.xlim((0, L))
    plt.ylim((-0.05, 1.05))
    plt.grid()
    plt.legend(handles=[total_line, opt_line], loc="best")
    plt.xlabel("slot index")
    plt.ylabel("throughput")
    plt.title("Total vs Optimal")
    plt.show()


if __name__ == "__main__":
    main()
