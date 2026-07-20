"""Linear layer building blocks."""

from torch import nn


class Linear(nn.Linear):
    """Ordinary single-device dense linear projection."""
