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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="rewards")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--save", type=str, default="results/throughput_coexist.png")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()

    agent_path = os.path.join(args.dir, "agent_coexist.txt")
    tdma_path = os.path.join(args.dir, "tdma_coexist.txt")
    aloha_path = os.path.join(args.dir, "aloha_coexist.txt")
    total_path = os.path.join(args.dir, "total_coexist.txt")

    if not os.path.exists(total_path):
        raise FileNotFoundError(f"missing file: {total_path}. Please run Run_coexist.py first.")

    total = np.loadtxt(total_path)
    L = len(total)

    total_ma = moving_avg(total, N=args.N)
    total_optimal = np.ones(L)

    plt.figure(figsize=(8, 4.8))
    line_total, = plt.plot(total_ma, lw=2.0, color="green", label="total throughput (MA)")
    line_opt, = plt.plot(total_optimal, lw=2.0, linestyle="--", color="black", label="upper bound = 1")

    # Optional component lines if files exist
    handles = [line_total, line_opt]
    if os.path.exists(agent_path):
        agent_ma = moving_avg(np.loadtxt(agent_path)[:L], N=args.N)
        line_agent, = plt.plot(agent_ma, lw=1.2, label="DRL throughput ")
        handles.append(line_agent)
    if os.path.exists(tdma_path):
        tdma_ma = moving_avg(np.loadtxt(tdma_path)[:L], N=args.N)
        line_tdma, = plt.plot(tdma_ma, lw=1.2, label="TDMA throughput ")
        handles.append(line_tdma)
    if os.path.exists(aloha_path):
        aloha_ma = moving_avg(np.loadtxt(aloha_path)[:L], N=args.N)
        line_aloha, = plt.plot(aloha_ma, lw=1.2, label="ALOHA throughput ")
        handles.append(line_aloha)

    plt.xlim((0, L))
    plt.ylim((-0.05, 1.05))
    plt.xlabel("slot index")
    plt.ylabel("throughput")
    plt.title("DRL MAC + TDMA + ALOHA coexistence")
    plt.grid(True)
    plt.legend(handles=handles, loc="best")
    plt.tight_layout()

    save_dir = os.path.dirname(args.save)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(args.save, dpi=180)
    print("saved figure:", args.save)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
