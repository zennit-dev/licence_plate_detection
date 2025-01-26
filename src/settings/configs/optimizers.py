from pydantic import BaseModel, Field


class AdamConfig(BaseModel):
    """Adam optimizer configuration settings."""

    learning_rate: float = Field(
        0.001, title="Learning rate", description="Learning rate for optimizer"
    )
    beta_1: float = Field(
        0.9, title="Beta 1", description="Exponential decay rate for the first moment"
    )
    beta_2: float = Field(
        0.999, title="Beta 2", description="Exponential decay rate for the second moment"
    )
    epsilon: float = Field(
        1e-7, title="Epsilon", description="Epsilon value for numerical stability"
    )
    amsgrad: bool = Field(False, title="AMSGrad", description="AMSGrad variant of this algorithm")


class SDGConfig(BaseModel):
    """Stochastic Gradient Descent optimizer configuration settings."""

    learning_rate: float = Field(
        0.01, title="Learning rate", description="Learning rate for optimizer"
    )
    momentum: float = Field(0.0, title="Momentum", description="Momentum for gradient descent")
    nesterov: bool = Field(False, title="Nesterov", description="Nesterov momentum")


class RMSPropConfig(BaseModel):
    """RMSProp optimizer configuration settings."""

    learning_rate: float = Field(
        0.001, title="Learning rate", description="Learning rate for optimizer"
    )
    rho: float = Field(0.9, title="Rho", description="Discounting factor for the history gradient")
    momentum: float = Field(0.0, title="Momentum", description="Momentum for gradient descent")
    epsilon: float = Field(
        1e-7, title="Epsilon", description="Epsilon value for numerical stability"
    )
    centered: bool = Field(
        False, title="Centered", description="Centered variant of this algorithm"
    )
