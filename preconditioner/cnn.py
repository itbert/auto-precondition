from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc

try:
    from .metrics import condition_number
    from .preconditioners import Preconditioner, PreconditionerFactory
except ImportError:
    from metrics import condition_number  # type: ignore
    from preconditioners import Preconditioner, PreconditionerFactory  # type: ignore


Array = NDArray[np.generic]


def torch_is_available() -> bool:
    return _TORCH_AVAILABLE


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for preconditioner.cnn. "
            "Install it first, for example with `pip install torch`."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class CNNInverseApproximatorConfig:
    matrix_size: int
    hidden_channels: int = 32
    dilations: tuple[int, ...] = (1, 2, 4, 2, 1)
    dropout: float = 0.0
    basis_rank: int = 12
    add_coordinate_channels: bool = True
    add_diagonal_channel: bool = True
    add_baseline_channels: bool = True
    use_jacobi_baseline: bool = True
    use_correction_line_search: bool = True
    epochs: int = 80
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    consistency_weight: float = 0.35
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.15
    early_stopping_patience: int | None = 20
    device: str | None = None
    seed: int = 0


@dataclass(frozen=True)
class CNNTrainingHistory:
    train_loss: tuple[float, ...]
    val_loss: tuple[float, ...]
    train_target_loss: tuple[float, ...]
    train_consistency_loss: tuple[float, ...]
    val_target_loss: tuple[float, ...]
    val_consistency_loss: tuple[float, ...]
    best_epoch: int


if _TORCH_AVAILABLE:

    def _group_count(channels: int) -> int:
        for groups in (8, 6, 4, 3, 2):
            if channels % groups == 0:
                return groups
        return 1


    class ResidualMatrixBlock(nn.Module):
        def __init__(
            self,
            channels: int,
            *,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            groups = _group_count(channels)
            padding = dilation
            self.norm1 = nn.GroupNorm(groups, channels)
            self.norm2 = nn.GroupNorm(groups, channels)
            self.act = nn.GELU()
            self.conv1 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            )
            self.conv2 = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=padding,
                dilation=dilation,
            )
            self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.conv1(self.act(self.norm1(x)))
            y = self.dropout(y)
            y = self.conv2(self.act(self.norm2(y)))
            return x + y


    class DownsampleBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)


    class MatrixCoefficientCNN(nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            hidden_channels: int,
            dilations: Sequence[int],
            dropout: float,
            output_dim: int,
        ) -> None:
            super().__init__()
            split = max(1, len(dilations) // 2)
            stage1_dilations = tuple(dilations[:split])
            stage2_dilations = tuple(dilations[split:]) or (1,)
            high_channels = hidden_channels * 2

            self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2)
            self.stage1 = nn.ModuleList(
                [
                    ResidualMatrixBlock(
                        hidden_channels,
                        dilation=int(max(1, dilation)),
                        dropout=dropout,
                    )
                    for dilation in stage1_dilations
                ]
            )
            self.downsample = DownsampleBlock(hidden_channels, high_channels)
            self.stage2 = nn.ModuleList(
                [
                    ResidualMatrixBlock(
                        high_channels,
                        dilation=int(max(1, dilation)),
                        dropout=dropout,
                    )
                    for dilation in stage2_dilations
                ]
            )
            groups = _group_count(high_channels)
            self.head = nn.Sequential(
                nn.GroupNorm(groups, high_channels),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(high_channels, high_channels),
                nn.GELU(),
                nn.Linear(high_channels, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.stem(x)
            for block in self.stage1:
                y = block(y)
            y = self.downsample(y)
            for block in self.stage2:
                y = block(y)
            return self.head(y)


class ConvolutionalInverseApproximator:
    """Learn a matrix inverse surrogate with a low-rank CNN correction model.

    The model uses:
    - a compact CNN encoder over the matrix treated as an image;
    - a low-rank decoder fitted from inverse/correction operators in the
      training set, which is much more stable for smooth matrix families;
    - an optional Jacobi baseline, so the network learns only the dense
      correction and can stay well behaved on real impedance matrices.
    """

    def __init__(
        self,
        *,
        matrix_size: int,
        hidden_channels: int = 32,
        dilations: Sequence[int] = (1, 2, 4, 2, 1),
        dropout: float = 0.0,
        basis_rank: int = 12,
        add_coordinate_channels: bool = True,
        add_diagonal_channel: bool = True,
        add_baseline_channels: bool = True,
        use_jacobi_baseline: bool = True,
        use_correction_line_search: bool = True,
        epochs: int = 80,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        consistency_weight: float = 0.35,
        gradient_clip_norm: float = 1.0,
        validation_fraction: float = 0.15,
        early_stopping_patience: int | None = 20,
        device: str | None = None,
        seed: int = 0,
    ) -> None:
        self.config = CNNInverseApproximatorConfig(
            matrix_size=int(matrix_size),
            hidden_channels=int(hidden_channels),
            dilations=tuple(int(max(1, value)) for value in dilations),
            dropout=float(dropout),
            basis_rank=int(basis_rank),
            add_coordinate_channels=bool(add_coordinate_channels),
            add_diagonal_channel=bool(add_diagonal_channel),
            add_baseline_channels=bool(add_baseline_channels),
            use_jacobi_baseline=bool(use_jacobi_baseline),
            use_correction_line_search=bool(use_correction_line_search),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            consistency_weight=float(consistency_weight),
            gradient_clip_norm=float(gradient_clip_norm),
            validation_fraction=float(validation_fraction),
            early_stopping_patience=early_stopping_patience,
            device=device,
            seed=int(seed),
        )
        self.matrix_size = self.config.matrix_size
        self._network: Any = None
        self._device: Any = None
        self._dtype: np.dtype = np.dtype(np.float64)
        self._data_channels = 0
        self._feature_data_channels = 0
        self._auxiliary_channels = 0
        self._basis_rank = 0
        self._feature_mean: Array | None = None
        self._feature_std: Array | None = None
        self._coefficient_mean: Array | None = None
        self._coefficient_std: Array | None = None
        self._basis_flat: Array | None = None
        self._target_center_flat: Array | None = None
        self._coordinate_channels: Array | None = None
        self.history: CNNTrainingHistory | None = None

    @classmethod
    def torch_available(cls) -> bool:
        return torch_is_available()

    @property
    def is_fitted(self) -> bool:
        return self._network is not None

    def fit(
        self,
        matrices: Sequence[Array],
        targets: Sequence[Array],
        *,
        validation_data: tuple[Sequence[Array], Sequence[Array]] | None = None,
        verbose: bool = False,
    ) -> "ConvolutionalInverseApproximator":
        _require_torch()
        if len(matrices) == 0 or len(targets) == 0:
            raise ValueError("training data must be non-empty")
        if len(matrices) != len(targets):
            raise ValueError("matrices and targets must have equal length")

        self._set_random_seed(self.config.seed)

        matrix_batch = self._stack_matrices(matrices)
        target_batch = self._stack_matrices(targets)
        self._dtype = np.dtype(np.result_type(matrix_batch.dtype, target_batch.dtype))
        self._data_channels = 2 if np.iscomplexobj(matrix_batch) or np.iscomplexobj(target_batch) else 1
        self._coordinate_channels = self._make_auxiliary_channels()
        self._auxiliary_channels = 0 if self._coordinate_channels is None else int(self._coordinate_channels.shape[0])

        baseline_batch = self._build_baseline_batch(matrix_batch)
        feature_data = self._build_feature_data(matrix_batch, baseline_batch)
        target_data = self._encode_data_channels(target_batch - baseline_batch)
        matrix_data = self._encode_data_channels(matrix_batch)
        baseline_data = self._encode_data_channels(baseline_batch)

        if validation_data is not None:
            if len(validation_data) != 2:
                raise ValueError("validation_data must be a tuple of (matrices, targets)")
            val_matrices, val_targets = validation_data
            if len(val_matrices) != len(val_targets):
                raise ValueError("validation matrices and targets must have equal length")
            if len(val_matrices) == 0:
                val_feature_data = None
                val_target_data = None
                val_matrix_data = None
                val_baseline_data = None
            else:
                val_matrix_batch = self._stack_matrices(val_matrices)
                val_target_batch = self._stack_matrices(val_targets)
                val_baseline_batch = self._build_baseline_batch(val_matrix_batch)
                val_feature_data = self._build_feature_data(val_matrix_batch, val_baseline_batch)
                val_target_data = self._encode_data_channels(val_target_batch - val_baseline_batch)
                val_matrix_data = self._encode_data_channels(val_matrix_batch)
                val_baseline_data = self._encode_data_channels(val_baseline_batch)
            train_feature_data = feature_data
            train_target_data = target_data
            train_matrix_data = matrix_data
            train_baseline_data = baseline_data
        else:
            (
                train_feature_data,
                train_target_data,
                train_matrix_data,
                train_baseline_data,
                val_feature_data,
                val_target_data,
                val_matrix_data,
                val_baseline_data,
            ) = self._split_train_validation(
                feature_data,
                target_data,
                matrix_data,
                baseline_data,
            )

        self._fit_feature_normalization(train_feature_data)
        train_coefficients = self._fit_target_basis(train_target_data)
        train_inputs = self._assemble_input_channels(train_feature_data)
        train_coefficients_norm = self._normalize_coefficients(train_coefficients)

        if val_feature_data is None or val_target_data is None or val_matrix_data is None or val_baseline_data is None:
            val_inputs = None
            val_coefficients_norm = None
        else:
            val_inputs = self._assemble_input_channels(val_feature_data)
            val_coefficients_norm = self._normalize_coefficients(
                self._project_target_data(val_target_data)
            )

        self._device = self._resolve_device()
        self._network = self._build_network()
        optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        train_loader = self._make_loader(
            train_inputs,
            train_coefficients_norm,
            train_matrix_data,
            train_baseline_data,
            shuffle=True,
        )
        val_loader = None
        if val_inputs is not None and val_coefficients_norm is not None:
            val_loader = self._make_loader(
                val_inputs,
                val_coefficients_norm,
                val_matrix_data,
                val_baseline_data,
                shuffle=False,
            )

        train_loss: list[float] = []
        train_target_loss: list[float] = []
        train_consistency_loss: list[float] = []
        val_loss: list[float] = []
        val_target_loss: list[float] = []
        val_consistency_loss: list[float] = []
        best_metric = float("inf")
        best_epoch = -1
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(self.config.epochs):
            metrics = self._run_epoch(train_loader, optimizer=optimizer)
            train_loss.append(metrics[0])
            train_target_loss.append(metrics[1])
            train_consistency_loss.append(metrics[2])

            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, optimizer=None)
                val_loss.append(val_metrics[0])
                val_target_loss.append(val_metrics[1])
                val_consistency_loss.append(val_metrics[2])
                monitored = val_metrics[0]
            else:
                val_loss.append(float("nan"))
                val_target_loss.append(float("nan"))
                val_consistency_loss.append(float("nan"))
                monitored = metrics[0]

            if monitored < best_metric:
                best_metric = monitored
                best_epoch = epoch
                best_state = deepcopy(self._network.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose and (epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == self.config.epochs):
                message = (
                    f"[CNNApprox] epoch {epoch + 1:>3d}/{self.config.epochs} "
                    f"train={metrics[0]:.5e}"
                )
                if val_loader is not None:
                    message += f" val={val_metrics[0]:.5e}"
                print(message)

            patience = self.config.early_stopping_patience
            if patience is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"[CNNApprox] early stopping at epoch {epoch + 1}; "
                        f"best epoch was {best_epoch + 1}"
                    )
                break

        if best_state is not None:
            self._network.load_state_dict(best_state)

        self.history = CNNTrainingHistory(
            train_loss=tuple(train_loss),
            val_loss=tuple(val_loss),
            train_target_loss=tuple(train_target_loss),
            train_consistency_loss=tuple(train_consistency_loss),
            val_target_loss=tuple(val_target_loss),
            val_consistency_loss=tuple(val_consistency_loss),
            best_epoch=max(best_epoch, 0),
        )
        return self

    def predict_inverse(self, A: Array) -> Array:
        self._check_fitted()
        matrix = np.asarray(A)
        baseline = self._build_baseline_matrix(matrix)
        feature_data = self._build_feature_data(
            self._stack_matrices([matrix]),
            self._stack_matrices([baseline]),
        )
        inputs = self._assemble_input_channels(feature_data)
        input_tensor = self._to_tensor(inputs)

        self._network.eval()
        with torch.no_grad():
            coefficients = self._network(input_tensor)
            correction = self._decode_coefficients_tensor(coefficients).detach().cpu().numpy()[0]

        correction_matrix = self._decode_single_operator(correction)
        operator = baseline + correction_matrix
        if self.config.use_jacobi_baseline and self.config.use_correction_line_search:
            beta = self._optimal_correction_scale(matrix, baseline, correction_matrix)
            operator = baseline + beta * correction_matrix
        return np.asarray(operator, dtype=self._dtype)

    def as_preconditioner(self, A: Array, *, name: str = "CNNApprox") -> Preconditioner:
        P = self.predict_inverse(A)

        def apply(x: NDArray[np.generic]) -> NDArray[np.generic]:
            return P @ x

        return Preconditioner(
            name=name,
            apply_vec=apply,
            size=self.matrix_size,
            matrix=P,
            dtype=P.dtype,
        )

    def describe_architecture(self) -> str:
        aux_terms: list[str] = []
        if self.config.add_coordinate_channels:
            aux_terms.append("row/column coordinates")
        if self.config.add_diagonal_channel:
            aux_terms.append("diagonal mask")
        if self.config.add_baseline_channels and self.config.use_jacobi_baseline:
            aux_terms.append("Jacobi residual channels")
        aux_text = ", ".join(aux_terms) if aux_terms else "no auxiliary channels"
        baseline_text = "Jacobi + low-rank CNN correction" if self.config.use_jacobi_baseline else "direct low-rank inverse"
        return (
            f"{baseline_text}: "
            f"hidden_channels={self.config.hidden_channels}, "
            f"dilations={self.config.dilations}, "
            f"basis_rank={self._basis_rank or self.config.basis_rank}, "
            f"aux={aux_text}, "
            f"consistency_weight={self.config.consistency_weight:.3f}"
        )

    def _check_fitted(self) -> None:
        if self._network is None:
            raise RuntimeError("model is not fitted")

    def _build_network(self) -> Any:
        _require_torch()
        if self._basis_rank <= 0:
            raise RuntimeError("target basis is not fitted")
        net = MatrixCoefficientCNN(
            in_channels=self._feature_data_channels + self._auxiliary_channels,
            hidden_channels=self.config.hidden_channels,
            dilations=self.config.dilations,
            dropout=self.config.dropout,
            output_dim=self._basis_rank,
        )
        return net.to(self._device)

    def _resolve_device(self) -> Any:
        _require_torch()
        if self.config.device is not None:
            return torch.device(self.config.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _make_loader(
        self,
        inputs: Array,
        coefficients: Array,
        matrix_data: Array,
        baseline_data: Array,
        *,
        shuffle: bool,
    ) -> Any:
        _require_torch()
        dataset = TensorDataset(
            self._to_tensor(inputs),
            self._to_tensor(coefficients),
            self._to_tensor(matrix_data),
            self._to_tensor(baseline_data),
        )
        return DataLoader(
            dataset,
            batch_size=min(self.config.batch_size, len(dataset)),
            shuffle=shuffle,
        )

    def _run_epoch(
        self,
        loader: Any,
        *,
        optimizer: Any | None,
    ) -> tuple[float, float, float]:
        _require_torch()
        if optimizer is None:
            self._network.eval()
        else:
            self._network.train()

        total_loss = 0.0
        total_target_loss = 0.0
        total_consistency_loss = 0.0
        total_samples = 0

        for batch_inputs, batch_coefficients, batch_matrix_data, batch_baseline_data in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(optimizer is not None):
                predicted_coefficients = self._network(batch_inputs)
                predicted_correction = self._decode_coefficients_tensor(predicted_coefficients)
                target_correction = self._decode_coefficients_tensor(batch_coefficients)
                target_loss = F.mse_loss(predicted_correction, target_correction)
                consistency_loss = self._consistency_loss(
                    predicted_correction,
                    batch_matrix_data,
                    batch_baseline_data,
                )
                loss = target_loss + self.config.consistency_weight * consistency_loss

            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(),
                    self.config.gradient_clip_norm,
                )
                optimizer.step()

            batch_size = int(batch_inputs.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_target_loss += float(target_loss.detach().cpu()) * batch_size
            total_consistency_loss += float(consistency_loss.detach().cpu()) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            total_loss / total_samples,
            total_target_loss / total_samples,
            total_consistency_loss / total_samples,
        )

    def _consistency_loss(
        self,
        predicted_correction: Any,
        matrix_data: Any,
        baseline_data: Any,
    ) -> Any:
        _require_torch()
        if self.config.consistency_weight == 0.0:
            return predicted_correction.new_zeros(())

        matrix_tensor = self._channels_to_matrix_tensor(matrix_data)
        operator_tensor = self._channels_to_matrix_tensor(baseline_data + predicted_correction)
        identity = torch.eye(
            self.matrix_size,
            device=predicted_correction.device,
            dtype=operator_tensor.dtype,
        ).unsqueeze(0)
        diff = operator_tensor @ matrix_tensor - identity
        if diff.is_complex():
            return torch.mean(diff.real.pow(2) + diff.imag.pow(2))
        return torch.mean(diff.pow(2))

    def _channels_to_matrix_tensor(self, channels: Any) -> Any:
        if self._data_channels == 1:
            return channels[:, 0, :, :]
        return torch.complex(channels[:, 0, :, :], channels[:, 1, :, :])

    def _to_tensor(self, array: Array) -> Any:
        _require_torch()
        return torch.as_tensor(array, dtype=torch.float32, device=self._device)

    def _stack_matrices(self, matrices: Sequence[Array]) -> Array:
        stacked = []
        for matrix in matrices:
            arr = np.asarray(matrix)
            if arr.shape != (self.matrix_size, self.matrix_size):
                raise ValueError(
                    f"Expected shape {(self.matrix_size, self.matrix_size)}, got {arr.shape}"
                )
            stacked.append(arr)
        return np.asarray(stacked)

    def _fit_feature_normalization(self, feature_data: Array) -> None:
        self._feature_data_channels = int(feature_data.shape[1])
        self._feature_mean = np.mean(feature_data, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self._feature_std = np.std(feature_data, axis=(0, 2, 3), keepdims=True).astype(np.float32)
        self._feature_std = np.where(self._feature_std > 1e-6, self._feature_std, 1.0)

    def _fit_target_basis(self, target_data: Array) -> Array:
        flat = target_data.reshape(len(target_data), -1).astype(np.float64, copy=False)
        center = np.mean(flat, axis=0, keepdims=True)
        centered = flat - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        self._basis_rank = min(self.config.basis_rank, vh.shape[0], vh.shape[1])
        basis = vh[: self._basis_rank].astype(np.float32, copy=False)
        coefficients = centered @ basis.T
        self._basis_flat = basis
        self._target_center_flat = center.astype(np.float32, copy=False)
        self._coefficient_mean = np.mean(coefficients, axis=0, keepdims=True).astype(np.float32)
        self._coefficient_std = np.std(coefficients, axis=0, keepdims=True).astype(np.float32)
        self._coefficient_std = np.where(self._coefficient_std > 1e-6, self._coefficient_std, 1.0)
        return coefficients.astype(np.float32, copy=False)

    def _project_target_data(self, target_data: Array) -> Array:
        if self._basis_flat is None or self._target_center_flat is None:
            raise RuntimeError("target basis is not fitted")
        flat = target_data.reshape(len(target_data), -1).astype(np.float32, copy=False)
        centered = flat - self._target_center_flat
        return centered @ self._basis_flat.T

    def _normalize_coefficients(self, coefficients: Array) -> Array:
        if self._coefficient_mean is None or self._coefficient_std is None:
            raise RuntimeError("coefficient normalization is not fitted")
        return ((coefficients - self._coefficient_mean) / self._coefficient_std).astype(np.float32)

    def _denormalize_coefficients_tensor(self, coefficients: Any) -> Any:
        mean = self._to_tensor(self._coefficient_mean)
        std = self._to_tensor(self._coefficient_std)
        return coefficients * std + mean

    def _decode_coefficients_tensor(self, coefficients: Any) -> Any:
        if self._basis_flat is None or self._target_center_flat is None:
            raise RuntimeError("target basis is not fitted")
        coefficients_denorm = self._denormalize_coefficients_tensor(coefficients)
        basis = self._to_tensor(self._basis_flat)
        center = self._to_tensor(self._target_center_flat)
        flat = coefficients_denorm @ basis + center
        return flat.view(-1, self._data_channels, self.matrix_size, self.matrix_size)

    def _decode_single_operator(self, channels: Array) -> Array:
        if self._data_channels == 1:
            operator = channels[0]
        else:
            operator = channels[0] + 1j * channels[1]
        return np.asarray(operator, dtype=self._dtype)

    def _encode_data_channels(self, matrices: Array) -> Array:
        matrices = np.asarray(matrices)
        if self._data_channels == 2:
            complex_batch = matrices.astype(np.complex128, copy=False)
            encoded = np.stack([complex_batch.real, complex_batch.imag], axis=1)
        else:
            encoded = matrices.astype(np.float64, copy=False)[:, None, :, :]
        return encoded.astype(np.float32, copy=False)

    def _build_feature_data(self, matrices: Array, baseline_batch: Array) -> Array:
        features = [self._encode_data_channels(matrices)]
        if self.config.add_baseline_channels and self.config.use_jacobi_baseline:
            identity = np.eye(self.matrix_size, dtype=matrices.dtype)[None, :, :]
            residual = baseline_batch @ matrices - identity
            features.append(self._encode_data_channels(residual))
        return np.concatenate(features, axis=1).astype(np.float32, copy=False)

    def _normalize_feature_data(self, feature_data: Array) -> Array:
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("feature normalization is not fitted")
        return ((feature_data - self._feature_mean) / self._feature_std).astype(np.float32)

    def _make_auxiliary_channels(self) -> Array | None:
        extra: list[Array] = []
        n = self.matrix_size
        if self.config.add_coordinate_channels:
            coords = np.linspace(-1.0, 1.0, num=n, dtype=np.float32)
            row = np.repeat(coords[:, None], n, axis=1)
            col = np.repeat(coords[None, :], n, axis=0)
            extra.extend([row, col])
        if self.config.add_diagonal_channel:
            extra.append(np.eye(n, dtype=np.float32))
        if not extra:
            return None
        return np.stack(extra, axis=0)

    def _assemble_input_channels(self, feature_data: Array) -> Array:
        normalized = self._normalize_feature_data(feature_data)
        if self._coordinate_channels is None:
            return normalized
        batch = normalized.shape[0]
        auxiliary = np.broadcast_to(
            self._coordinate_channels[None, ...],
            (batch, *self._coordinate_channels.shape),
        ).astype(np.float32)
        return np.concatenate([normalized, auxiliary], axis=1)

    def _split_train_validation(
        self,
        feature_data: Array,
        target_data: Array,
        matrix_data: Array,
        baseline_data: Array,
    ) -> tuple[Array, Array, Array, Array, Array | None, Array | None, Array | None, Array | None]:
        if len(feature_data) <= 1 or self.config.validation_fraction == 0.0:
            return feature_data, target_data, matrix_data, baseline_data, None, None, None, None
        rng = np.random.default_rng(self.config.seed)
        indices = rng.permutation(len(feature_data))
        val_count = int(round(len(feature_data) * self.config.validation_fraction))
        val_count = max(1, min(len(feature_data) - 1, val_count))
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        return (
            feature_data[train_idx],
            target_data[train_idx],
            matrix_data[train_idx],
            baseline_data[train_idx],
            feature_data[val_idx],
            target_data[val_idx],
            matrix_data[val_idx],
            baseline_data[val_idx],
        )

    def _build_baseline_matrix(self, matrix: Array) -> Array:
        if not self.config.use_jacobi_baseline:
            return np.zeros_like(matrix, dtype=self._dtype)
        diag = np.diag(matrix)
        inv_diag = np.where(np.abs(diag) > 1e-12, 1.0 / diag, 1.0)
        baseline = np.zeros((self.matrix_size, self.matrix_size), dtype=np.result_type(matrix.dtype, np.complex128))
        baseline[np.arange(self.matrix_size), np.arange(self.matrix_size)] = inv_diag
        return baseline

    def _build_baseline_batch(self, matrices: Array) -> Array:
        if not self.config.use_jacobi_baseline:
            return np.zeros_like(matrices, dtype=self._dtype)
        baselines = np.zeros_like(matrices, dtype=np.result_type(matrices.dtype, np.complex128))
        diag = np.diagonal(matrices, axis1=1, axis2=2)
        inv_diag = np.where(np.abs(diag) > 1e-12, 1.0 / diag, 1.0)
        idx = np.arange(self.matrix_size)
        baselines[:, idx, idx] = inv_diag
        return baselines

    def _optimal_correction_scale(self, matrix: Array, baseline: Array, correction: Array) -> complex:
        residual = baseline @ matrix - np.eye(self.matrix_size, dtype=np.result_type(matrix.dtype, correction.dtype))
        direction = correction @ matrix
        denom = np.vdot(direction, direction)
        if abs(denom) <= 1e-12:
            return 0.0
        return -np.vdot(direction, residual) / denom

    def _set_random_seed(self, seed: int) -> None:
        np.random.seed(seed)
        if _TORCH_AVAILABLE:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


def build_cnn_preconditioner_factory(
    model: ConvolutionalInverseApproximator,
    *,
    name: str = "CNNApprox",
    notes: str = "Convolutional low-rank inverse approximator",
) -> PreconditionerFactory:
    return PreconditionerFactory(
        name=name,
        builder=lambda A, trained_model=model, model_name=name: trained_model.as_preconditioner(
            A,
            name=model_name,
        ),
        max_n=model.matrix_size,
        notes=notes,
    )


def evaluate_cnn_inverse_model(
    model: ConvolutionalInverseApproximator,
    matrices: Iterable[Array],
    targets: Iterable[Array],
    *,
    signed_kappa: bool = True,
) -> tuple[float, float, float, float]:
    rel_errors: list[float] = []
    kappas: list[float] = []
    for A, T in zip(matrices, targets):
        P = model.predict_inverse(A)
        rel_errors.append(
            float(np.linalg.norm(P - T, ord="fro") / max(np.linalg.norm(T, ord="fro"), 1e-12))
        )
        kappas.append(
            float(condition_number(P @ A, signed=signed_kappa and not np.iscomplexobj(P @ A)))
        )
    return (
        float(np.mean(rel_errors)) if rel_errors else float("nan"),
        float(np.std(rel_errors)) if rel_errors else float("nan"),
        float(np.mean(kappas)) if kappas else float("nan"),
        float(np.std(kappas)) if kappas else float("nan"),
    )


__all__ = [
    "CNNInverseApproximatorConfig",
    "CNNTrainingHistory",
    "ConvolutionalInverseApproximator",
    "build_cnn_preconditioner_factory",
    "evaluate_cnn_inverse_model",
    "torch_is_available",
]
