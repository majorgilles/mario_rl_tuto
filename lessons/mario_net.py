"""Convolutional action-value network for the Mario DDQN agent."""

from typing import Literal

import torch
from torch import nn


class MarioNet(nn.Module):
    """Predict action values from a stacked Mario observation.

    Args:
        input_dim: State shape in (frames, height, width) order. The current
            environment provides (4, 84, 84).
        output_dim: Number of discrete actions to score. The current environment
            provides 2 actions: right, or right while jumping.
    """

    def __init__(
        self,
        input_dim: tuple[int, int, int],
        output_dim: int,
    ) -> None:
        super().__init__()
        channels, height, width = input_dim

        if height != 84 or width != 84:
            raise ValueError(f"Expected 84 x 84 input frames, got {height} x {width}.")

        self.online = self._build_cnn(channels, output_dim)
        self.target = self._build_cnn(channels, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for parameter in self.target.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        state: torch.Tensor,
        model: Literal["online", "target"],
    ) -> torch.Tensor:
        """Return Q-value predictions for every action in each input state.

        Args:
            state: Batch of processed observations shaped
                (batch_size, frames, height, width). In this notebook, each
                state has shape (4, 84, 84), so a one-state batch is
                (1, 4, 84, 84). The four frames are input channels that
                provide short-term motion information.
            model: Which DDQN network makes the prediction: ``online`` learns
                through gradients, while ``target`` stays fixed between syncs.

        Returns:
            A tensor shaped (batch_size, action_dim). Each element [i, j] is
            a Q value: the network's predicted discounted future reward for
            taking action j from state i, then continuing to act.
        """
        if model == "online":
            return self.online(state)
        if model == "target":
            return self.target(state)
        raise ValueError(f"Unknown model: {model!r}")

    @staticmethod
    def _build_cnn(channels: int, output_dim: int) -> nn.Sequential:
        """Build one convolutional network that maps frames to Q values.

        Convolutions scan nearby pixels with shared filters, which is useful
        for finding visual patterns such as edges, platforms, and Mario. The
        strides also reduce the spatial size: 84 x 84 becomes 20 x 20, then
        9 x 9, then 7 x 7. At that point there are 64 feature maps, so
        flattening produces 64 * 7 * 7 = 3136 values. Dense layers combine
        those visual features and emit one score for each possible action.

        Args:
            channels: Number of input frame channels. The current frame stack
                provides 4 channels.
            output_dim: Number of action scores to produce. The current
                JoypadSpace action set has 2 actions.

        Returns:
            A sequential convolutional network mapping a state batch shaped
            (batch_size, 4, 84, 84) to Q values shaped (batch_size, 2).
        """
        return nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
