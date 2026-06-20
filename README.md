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

This project uses `uv` and targets Python 3.10 because the tutorial depends on legacy Gym-era packages.

```bash
uv python install 3.10
uv sync
uv run python -m ipykernel install --user --name mario-rl-tuto --display-name "Python (mario-rl-tuto)"
uv run jupyter notebook
```

For CUDA-enabled PyTorch, follow the install command recommended by [pytorch.org](https://pytorch.org/get-started/locally/) for your GPU/driver, then update this environment as needed.

> Note: the source tutorial uses legacy `gym`, `gym-super-mario-bros`, `tensordict==0.3.0`, and `torchrl==0.3.0`. `numpy<2` is pinned for compatibility with unmaintained Gym packages.

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
