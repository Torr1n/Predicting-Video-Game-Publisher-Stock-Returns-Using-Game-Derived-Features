import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from datetime import datetime, timedelta

from forecasting.utils import standardize_cols


class PurgedKFold:
    """
    Implements Combinatorial Purged K-Fold Cross Validation to prevent forward-looking bias.
    This custom implementation is necessary as it's specific to time series data and not
    available in standard sklearn.
    """

    def __init__(self, n_splits=5, purge_window=timedelta(weeks=2)):
        self.n_splits = n_splits
        self.purge_window = purge_window

    def split(self, X, dates):
        # Convert dates to datetime if they aren't already
        dates = pd.to_datetime(dates)
        indices = np.arange(len(dates))

        # Create basic k-fold splits
        kf = KFold(n_splits=self.n_splits, shuffle=False)

        for train_idx, val_idx in kf.split(X):
            # Get the validation set dates
            val_dates = dates[val_idx]

            # Purge training samples that are too close to validation samples
            purged_train_idx = []
            for idx in train_idx:
                date = dates[idx]
                # Check if this training sample is outside the purge window
                if not any(
                    (date >= val_date - self.purge_window)
                    & (date <= val_date + self.purge_window)
                    for val_date in val_dates
                ):
                    purged_train_idx.append(idx)

            yield np.array(purged_train_idx), val_idx


def evaluate_predictions(
    y_train, y_train_pred, w_train, y_val, y_pred, w_val=None, verbose=False
):
    """
    Calculate and optionally print evaluation metrics.
    Returns dict of metrics for further analysis.
    """

    def weighted_loss(y_true, y_pred, sample_weight=None):
        # Binary weighted classification error
        if sample_weight is None:
            sample_weight = np.ones_like(y_true)
        incorrect = (y_true != y_pred).astype(float)
        return np.average(incorrect, weights=sample_weight)

    # Calculate metrics including both validation and training loss
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred, sample_weight=w_val),
        "precision": precision_score(y_val, y_pred, sample_weight=w_val),
        "recall": recall_score(y_val, y_pred, sample_weight=w_val),
        "f1": f1_score(y_val, y_pred, sample_weight=w_val),
        "validation_loss": weighted_loss(y_val, y_pred, sample_weight=w_val),
        "training_loss": weighted_loss(y_train, y_train_pred, sample_weight=w_train),
    }

    if verbose:
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Validation Loss: {metrics['validation_loss']:.4f}")
        print(f"Training Loss: {metrics['training_loss']:.4f}")

    return metrics


def run_cv_evaluation(
    model_fn,
    X,
    y,
    dates,
    weights=None,
    cv=5,
    purge_window=timedelta(weeks=8),
    verbose=False,
    kernel=None,
    standardize=False,
):
    """
    Runs cross-validation evaluation using PurgedKFold.

    Args:
        model_fn: Function that returns a fresh model instance
        X: Feature matrix
        y: Labels
        dates: Dates for purging
        weights: Sample weights
        cv: Number of folds
        purge_window: Time window for purging
        verbose: Whether to print detailed metrics
    """
    cv_splitter = PurgedKFold(n_splits=cv, purge_window=purge_window)
    cv_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, dates), 1):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train = weights[train_idx] if weights is not None else None
        w_val = weights[val_idx] if weights is not None else None
        if kernel:
            X_val = kernel(X_val, X)
            X_train = kernel(X_train, X)
        if standardize:
            X_train, mu, sigma = standardize_cols(X_train)
            X_val, *_ = standardize_cols(X_val, mu, sigma)
        # Get fresh model and fit
        model = model_fn()
        model.fit(X_train, y_train, sample_weight=w_train)
        y_pred = model.predict(X_val)
        print(f"labels {y_val}")
        print(f"predicitons: {y_pred}")
        y_train_pred = model.predict(X_train)
        # Evaluate
        fold_metrics = evaluate_predictions(
            y_train, y_train_pred, w_train, y_val, y_pred, w_val
        )
        cv_metrics.append(fold_metrics)

        if verbose:
            print(f"\nFold {fold} Results:")
            for metric, value in fold_metrics.items():
                print(f"{metric}: {value:.4f}")

    # Calculate aggregate statistics
    mean_metrics = {k: np.mean([m[k] for m in cv_metrics]) for k in cv_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in cv_metrics]) for k in cv_metrics[0]}

    if verbose:
        print("\nOverall CV Results:")
        for metric in mean_metrics:
            print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    return {"mean": mean_metrics, "std": std_metrics}
