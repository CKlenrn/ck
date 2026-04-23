# DR-DLMA Simulation Code

This repository contains simulation code for DR-DLMA-style deep reinforcement learning medium access control experiments, including TDMA, ALOHA, and coexistence scenarios.

## Contents

- `Brain.py`: DQN implementation based on Keras/TensorFlow.
- `env.py`, `env_aloha.py`, `env_coexist*.py`: simulation environments.
- `Run.py`, `Run_aloha.py`, `Run_coexist*.py`: training and simulation entry points.
- `throughput*.py`: throughput plotting scripts.
- `sweep_delay_*.py`: delay sweep experiment scripts.
- `rewards/`: saved reward/throughput samples.
- `results/`: generated result figures and summaries.

## Requirements

The original code uses the TensorFlow 1.x Keras backend API.

```bash
pip install -r requirements.txt
```

If you use a modern Python environment, TensorFlow 1.x may require an older Python version such as Python 3.6 or 3.7.

## Usage

Run the main DR-DLMA simulation:

```bash
python Run.py
```

Run the ALOHA scenario:

```bash
python Run_aloha.py
```

Run the coexistence scenario:

```bash
python Run_coexist.py --max_iter 20000
```

Generate a coexistence throughput figure after running the coexistence simulation:

```bash
python throughput_coexist.py --no_show
```

## Notes

- Generated Python cache files and local IDE settings are ignored.
- Some scripts write output files under `rewards/` and `results/`.
- Please adjust GPU settings in `Brain.py` if your machine uses a different CUDA device configuration.

## License

This project is released under the MIT License. See `LICENSE` for details.
