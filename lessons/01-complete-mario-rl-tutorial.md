# Lesson 01 — Complete the PyTorch Mario RL Tutorial

> ## Lesson link
> [Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

- **Duration/pages:** Single PyTorch intermediate tutorial
- **Course/book:** PyTorch Tutorials
- **Reading guide:** [../readings/01-complete-mario-rl-tutorial.md](../readings/01-complete-mario-rl-tutorial.md)
- **Workspace:** [../coursework/01-complete-mario-rl-tutorial/](../coursework/01-complete-mario-rl-tutorial/)
- **GitHub issue:** [#1](https://github.com/majorgilles/mario_rl_tuto/issues/1)

## Focus

Build and understand a Mario-playing reinforcement learning agent using PyTorch and Double Deep Q-Networks (DDQN). Use the original tutorial as the source of truth, and keep this repo for your own notes, experiments, and reproducibility artifacts.

## Learning checkpoint

After this unit, you should be able to:

- Explain the roles of environment, action, state, reward, return, and Q-value.
- Initialize the Super Mario Bros Gym environment and constrain the action space.
- Preprocess image observations with frame skipping, grayscale conversion, resizing, and frame stacking.
- Implement an epsilon-greedy agent with replay memory.
- Explain why DDQN uses online and target Q-networks.
- Track training progress with reward, episode length, loss, and Q-value metrics.

## Work plan

- [ ] Read the RL definitions and write a short glossary in your notes.
- [ ] Set up the environment from `pyproject.toml` with `uv sync`.
- [ ] Reproduce the environment initialization and wrapper pipeline.
- [ ] Implement or adapt the `Mario` agent methods: `act`, `cache`, `recall`, and `learn`.
- [ ] Implement the DDQN network, TD estimate, TD target, optimizer step, target sync, and checkpoint save.
- [ ] Run a short smoke-test training loop.
- [ ] Save any representative plots or metrics you want to keep.
- [ ] Update the reading guide and workspace README with takeaways and gotchas.
