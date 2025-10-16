import torch


class NormalizationModule(torch.nn.Module):
    def __init__(self):
        super(NormalizationModule, self).__init__()

    def is_scripted(self) -> bool:
        return isinstance(self, torch.jit.ScriptModule)

    def save_as_scripted(self, path: str) -> None:
        scripted = self
        if not self.is_scripted():
            scripted = torch.jit.script(scripted)
        scripted.save(path)


# [0; 1]
class MinMaxNormalizationModule(NormalizationModule):
    def __init__(self):
        super(MinMaxNormalizationModule, self).__init__()
         # Use register_buffer for non-trainable model state
        self.register_buffer('_x_min', None)
        self.register_buffer('_x_max', None)

    def fit(self, X_train: torch.Tensor) -> None:
        self._x_min = X_train.amin(dim=0, keepdim=True)
        self._x_max = X_train.amax(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._x_min) / (self._x_max - self._x_min)
        return x


# [-1; 1] (preserves 0s which maybe good)
class MaxAbsNormalizationModule(NormalizationModule):
    def __init__(self):
        super(MaxAbsNormalizationModule, self).__init__()
        self.register_buffer('_x_max', None)

    def fit(self, X_train: torch.Tensor) -> None:
        self._x_max = X_train.amax(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.abs(self._x_max)
        return x


# this Normalization technique was used in the TD3+BC paper
class StandardNormalizationModule(NormalizationModule):
    def __init__(self, eps: float = 1e-3):
        super(StandardNormalizationModule, self).__init__()
        self.register_buffer('_x_mean', None)
        self.register_buffer('_x_std', None)
        self._eps = eps

    def fit(self, X_train: torch.Tensor) -> None:
        self._x_mean = X_train.mean(dim=0, keepdim=True)
        self._x_std = X_train.std(dim=0, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._x_mean) / (self._x_std + self._eps)
        return x


# works well with outliers (maybe more optimal for Replay Buffer?)
class RobustNormalizationModule(NormalizationModule):
    def __init__(self):
        super(RobustNormalizationModule, self).__init__()
        self.register_buffer("_x_median", None)
        self.register_buffer("_iqr", None)

    def fit(self, X_train: torch.Tensor) -> None:
        q1 = X_train.quantile(0.25, dim=0, keepdim=True)
        q3 = X_train.quantile(0.75, dim=0, keepdim=True)

        self._iqr = q3 - q1
        # to avoid 0-division
        self._iqr[self._iqr == 0] = 1.0
        # required since .median returns (values, indices)
        self._x_median = X_train.median(dim=0, keepdim=True).values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._x_median) / self._iqr
