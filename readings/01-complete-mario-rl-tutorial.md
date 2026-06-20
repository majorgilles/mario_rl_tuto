# Reading Guide 01 — Train a Mario-playing RL Agent

Source: [PyTorch tutorial: Train a Mario-playing RL Agent](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

This file is a summary and study guide only. It intentionally avoids copying the full tutorial text or full source code.

## Summary

- The tutorial introduces core reinforcement learning concepts and applies them to a Super Mario Bros environment.
- The environment is reduced to a small action space: move right, or move right and jump.
- Raw RGB frames are preprocessed into stacked grayscale `84x84` frames so the agent can infer motion while keeping the state compact.
- The `Mario` agent uses epsilon-greedy exploration, replay memory, and a convolutional Q-network.
- Learning follows Double DQN: an online network selects the best next action, while a target network estimates the target Q-value.
- Training logs reward, episode length, loss, and Q-values, and periodically saves checkpoints.

## Section map

1. **RL definitions** — environment, action, state, reward, return, and optimal action-value function.
2. **Environment** — initialize `gym-super-mario-bros`, wrap it with `JoypadSpace`, and inspect `env.step()` output.
3. **Preprocessing** — apply `SkipFrame`, grayscale conversion, resizing, and `FrameStack`.
4. **Agent** — define Mario's responsibilities: act, remember, recall, and learn.
5. **Act** — use epsilon-greedy action selection and decay exploration over time.
6. **Cache and recall** — store experiences in a replay buffer and sample batches for learning.
7. **Learn** — implement DDQN with online/target networks, TD estimates, TD targets, optimizer updates, and target sync.
8. **Logging** — record episode-level reward, length, loss, and Q-value statistics.
9. **Training loop** — run short demo training, while recognizing meaningful training requires many more episodes.
10. **Conclusion** — the same method can be adapted to other Gym-style environments.

## Implementation checklist

- [ ] Confirm package compatibility before running long training.
- [ ] Create the Mario environment and verify one step returns observation, reward, done/truncation flags, and info.
- [ ] Verify wrapped observations have shape `(4, 84, 84)`.
- [ ] Confirm replay buffer samples tensors on the expected device.
- [ ] Test the CNN forward pass with a dummy batch.
- [ ] Run a tiny training smoke test before attempting long runs.
- [ ] Keep checkpoints outside git unless a small model artifact is intentionally curated.

## Gotchas

- The tutorial uses legacy `gym`; modern projects often use `gymnasium`, but this source tutorial is written for the older API.
- `numpy<2` is pinned because unmaintained Gym packages can break on NumPy 2.x.
- Short demo runs are not expected to learn a strong policy. The tutorial notes that meaningful Mario training may require tens of thousands of episodes.
- Checkpoints and videos can become large quickly; keep them in ignored local folders unless you intentionally publish small artifacts.
- If using CUDA, install the PyTorch build that matches your driver and platform.

## Notes

Use this section for your own observations while following the tutorial.

- 
