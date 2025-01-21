import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional


class NeuralNetWrapper:
    """
    Wrapper class to make PyTorch neural networks compatible with sklearn-style API.
    Handles training loop and prediction logic.
    """

    def __init__(
        self,
        model,
        optimizer_class,
        learning_rate,
        epochs=100,
        batch_size=32,
    ):
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.batch_size = batch_size

    def _transform_labels(self, y):
        """Transform [-1, 1] labels to [0, 1] for BCE loss"""
        return (y + 1) / 2

    def _inverse_transform_labels(self, y):
        """Transform [0, 1] predictions back to [-1, 1] format"""
        return 2 * y - 1

    def fit(self, X, y, sample_weight=None):
        # Convert numpy arrays to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_transformed = self._transform_labels(y)
        y_tensor = torch.FloatTensor(y_transformed).reshape(-1, 1)
        # Handle sample weights - expecting we will always have them though
        if sample_weight is not None:
            weight_tensor = torch.FloatTensor(sample_weight).reshape(-1, 1)
        else:
            print("No weights!")
        # Create data loader for mini-batch training
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor, weight_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y, batch_weights in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                criterion = nn.BCEWithLogitsLoss(weight=batch_weights, reduction="mean")
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            predictions = self._inverse_transform_labels(predictions)
        return predictions.numpy().reshape(-1)

    def predict_proba(self, X):
        """
        Generate prediction probabilities using sigmoid activation.
        """
        X_tensor = torch.FloatTensor(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.sigmoid(outputs)
        return probabilities.numpy().reshape(-1)


class LSTMPredictor(nn.Module):
    """
    LSTM-based model for stock return prediction.

    Architecture:
    - LSTM layers with optional dropout
    - Dense output layer
    - Configurable hidden dimensions and number of layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # x shape: (batch_size, seq_length, features)
        lstm_out, hidden = self.lstm(x, hidden)

        # Use only the last output for prediction
        last_output = lstm_out[:, -1, :]

        # Apply dropout and dense layer
        out = self.dropout(last_output)
        out = self.dense(out)
        return out


class InverseSquareRootScheduler:
    """
    Implements inverse square root learning rate scheduling

    The learning rate at step t is computed as:
    lr = base_lr * min(1/sqrt(t)

    Args:
        optimizer: PyTorch optimizer
        base_lr: Base learning rate
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 0.1,
        min_lr: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        # Inverse square root decay
        lr = self.base_lr * 1.0 / np.sqrt(self.current_step)

        # Apply minimum learning rate
        lr = max(self.min_lr, lr)

        # Update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


class LSTMWrapper:
    """
    Wrapper class to make PyTorch LSTM compatible with sklearn-style API.
    Handles sequence creation, training loop, and prediction logic with learning rate scheduling.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 10,
        base_lr: float = 0.1,
        min_lr: float = 1e-5,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.model = None

    def _create_sequences(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for LSTM input maintaining temporal order."""
        sequences = []
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i : i + self.sequence_length])
        return torch.FloatTensor(np.array(sequences))

    def _transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Transform [-1, 1] labels to [0, 1] for BCE loss"""
        return (y + 1) / 2

    def _inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Transform [0, 1] predictions back to [-1, 1] format"""
        return 2 * y - 1

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None
    ):
        # Initialize model if not exists
        if self.model is None:
            self.model = LSTMPredictor(
                input_dim=X.shape[1],
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )

        # Create sequences
        X_seq = self._create_sequences(X)
        y = y[self.sequence_length :]  # Align labels with sequences
        y_transformed = self._transform_labels(y)

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = sample_weight[self.sequence_length :]
            weight_tensor = torch.FloatTensor(sample_weight).reshape(-1, 1)
        else:
            weight_tensor = torch.ones(len(y)).reshape(-1, 1)

        # Convert to tensors
        y_tensor = torch.FloatTensor(y_transformed).reshape(-1, 1)

        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(X_seq, y_tensor, weight_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Maintain temporal order
        )

        # Initialize optimizer, scheduler, and criterion
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.base_lr)
        scheduler = InverseSquareRootScheduler(
            optimizer,
            base_lr=self.base_lr,
            min_lr=self.min_lr,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        # Training loop with early stopping
        best_loss = float("inf")
        patience_counter = 0
        learning_rates = []  # Track learning rates for analysis

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0

            for batch_X, batch_y, batch_weights in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                weighted_loss = (loss * batch_weights).mean()
                weighted_loss.backward()
                optimizer.step()
                lr = scheduler.step()
                learning_rates.append(lr)
                epoch_loss += weighted_loss.item()

            avg_loss = epoch_loss / len(loader)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions maintaining temporal order."""
        self.model.eval()
        X_seq = self._create_sequences(X)

        with torch.no_grad():
            outputs = self.model(X_seq)
            predictions = torch.sigmoid(outputs) > 0.5
            predictions = self._inverse_transform_labels(predictions.numpy())

        # Pad predictions to match input length
        padded_predictions = np.zeros(len(X))
        padded_predictions[self.sequence_length :] = predictions.reshape(-1)
        padded_predictions[: self.sequence_length] = predictions[0]  # Forward fill

        return padded_predictions
