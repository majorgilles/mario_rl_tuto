"""Mario DDQN agent with epsilon-greedy actions and replay memory."""

from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from mario_net import MarioNet


class Mario:
    """Select Mario actions and store replay-memory experiences.

    Args:
        state_dim: Shape of one processed state in (frames, height, width) order.
            In this notebook it is (4, 84, 84): four consecutive grayscale
            frames, each 84 pixels high and 84 pixels wide.
        action_dim: Number of discrete actions Mario may choose. With the current
            JoypadSpace mapping it is 2: right, or right while jumping.
        save_dir: Directory where later tutorial sections save MarioNet checkpoints.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        save_dir: Path,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = MarioNet(self.state_dim, self.action_dim).float().to(self.device)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        self.save_every = 500_000

        self.memory = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(
                100_000,
                device=torch.device("cpu"),
            )
        )
        self.batch_size = 32

    def act(self, state: np.ndarray) -> int:
        """Choose a random or network-selected action, then update epsilon.

        Args:
            state: Current processed observation with shape (4, 84, 84). The four
                frames give the network short-term movement information.

        Returns:
            The selected action index: 0 for right or 1 for right while jumping.
        """
        if np.random.random() < self.exploration_rate:
            action_idx = int(np.random.randint(self.action_dim))
        else:
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            action_values = self.net(state_tensor, model="online")
            action_idx = int(torch.argmax(action_values, dim=1).item())

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min,
            self.exploration_rate,
        )
        self.curr_step += 1
        return action_idx

    def cache(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: int,
        reward: float,
        done: bool,
    ) -> None:
        """Store one transition in replay memory.

        Args:
            state: Frames before the action, shaped (4, 84, 84).
            next_state: Frames after the action, shaped (4, 84, 84).
            action: Selected action index: 0 for right or 1 for right plus jump.
            reward: Feedback received after the action.
            done: Whether the episode ended after the action.
        """
        experience = TensorDict(
            {
                "state": torch.as_tensor(state, dtype=torch.float32),
                "next_state": torch.as_tensor(next_state, dtype=torch.float32),
                "action": torch.tensor([action]),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done]),
            },
            batch_size=[],
        )
        self.memory.add(experience)

    def recall(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random experience batch and move it to the model device.

        Returns:
            State, next-state, action, reward, and done tensors for one batch.
        """
        batch = self.memory.sample(self.batch_size).to(self.device)
        return (
            batch["state"],
            batch["next_state"],
            batch["action"].squeeze(),
            batch["reward"].squeeze(),
            batch["done"].squeeze(),
        )
