# Mario RL Tutorial Follow-Along

Follow-along repository for the PyTorch tutorial [Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html).

**Original tutorial authors:** Yuansong Feng, Suraj Subramanian, Howard Wang, and Steven Guo.  
**Original reference implementation:** [yuansongFeng/MadMario](https://github.com/yuansongFeng/MadMario/).  
**Tutorial created:** 2020-12-17. **Last updated:** 2024-02-05.

This repo is a personal learning workspace for completing the tutorial end-to-end. It intentionally stores summaries, links, notes, and starter workspaces only; it does **not** republish the full PyTorch tutorial text or source code.

## Source links

- [PyTorch tutorial: Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)
- [Double Deep Q-Networks paper](https://arxiv.org/pdf/1509.06461.pdf)
- [OpenAI Spinning Up: RL introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [RL cheatsheet linked by the tutorial](https://colab.research.google.com/drive/1eN33dPVtdPViiS1njTW_-r-IYCDTFU7N)
- [Original full code repository](https://github.com/yuansongFeng/MadMario/)

## Setup

This project uses `uv` and targets **Python 3.13**. It runs the modernized stack
(`gymnasium`, `gym-super-mario-bros>=9.1`, `nes-py>=9.0.1`) so that `nes-py` installs
from a prebuilt Windows wheel instead of compiling C++ from source.

```bash
uv python install 3.13
uv sync --no-install-package torchvision
uv run python -m ipykernel install --user --name mario-rl-tuto --display-name "Python (mario-rl-tuto)"
uv run jupyter notebook
```

### CUDA on Windows (torch + torchvision)

`torch` is pinned to the CUDA 12.6 build via the `pytorch-cu126` index in `pyproject.toml`.
The PyTorch index does **not** publish hashes for its Windows torchvision CUDA wheels,
which makes plain `uv sync` fail its hash check on torchvision. Work around it by
excluding torchvision from `uv sync` (above) and installing it separately:

```bash
# after `uv sync --no-install-package torchvision`:
uv pip install "torchvision>=0.22" --index https://download.pytorch.org/whl/cu126 --no-verify-hashes
```

Verify the GPU is picked up (torch and torchvision must both report `+cu126`):

```bash
uv run --no-sync python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
# -> 2.13.0+cu126 0.28.0+cu126 True
```

> **Important:** a plain `uv sync` (without `--no-install-package torchvision`) will fail on
> the torchvision hash and can revert `torch` to the CPU build. If `torch.cuda.is_available()`
> ever returns `False` or you hit `operator torchvision::nms does not exist`, re-run the two
> commands above to restore the matched CUDA builds.

> Note: the original tutorial used legacy `gym`, `gym-super-mario-bros==7.4.0`,
> `tensordict==0.3.0`, and `torchrl==0.3.0`. This repo modernizes to `gymnasium`,
> `torchrl>=0.9`, and `tensordict>=0.9`; the notebook is adapted to the Gymnasium API
> (`env.step()` returns 5 values, `env.reset()` returns `(obs, info)`).

## Lesson table

| # | Unit | Source | Notes | Reading guide | Workspace | Issue |
|---|------|--------|-------|---------------|-----------|-------|
| 01 | Complete the PyTorch Mario RL tutorial | [tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html) | [lesson](lessons/01-complete-mario-rl-tutorial.md) | [guide](readings/01-complete-mario-rl-tutorial.md) | [workspace](coursework/01-complete-mario-rl-tutorial/) | [#1](https://github.com/majorgilles/mario_rl_tuto/issues/1) |

## What you will build

- A Super Mario Bros environment with a restricted action space.
- Observation preprocessing wrappers: frame skip, grayscale, resize, and frame stack.
- A Mario agent that acts with epsilon-greedy exploration.
- Replay-buffer based experience caching and sampling.
- A Double DQN model with online and target networks.
- Training loop logging, checkpointing, and plots for reward/loss/Q metrics.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── .python-version
├── lessons/              # lightweight lesson notes
├── readings/             # summary/reading guide, not copied tutorial text
├── coursework/           # follow-along workspace
├── data/                 # optional local data, ignored by default in raw folders
└── artifacts/            # curated outputs worth keeping
```

## Copyright and attribution

All tutorial content belongs to its original authors and the PyTorch documentation project. Keep this repository limited to your own notes, summaries, links, and original follow-along code/artifacts.
