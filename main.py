from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from forecasting.RBF_kernels import GaussianRBFKernel
from feature_engineering.feature_engineering import *
from forecasting.utils import (
    ReportFormatter,
    create_TI_features,
    create_lagged_features,
    handle,
    merge_features,
    prepare_ml_arrays,
    run_feature_diagnostics,
    savefig,
    standardize_cols,
    main,
)
from portfolio import GamePublisherBLPortfolio, PortfolioReportFormatter
from publisher import *
from forecasting.purged_CV import evaluate_predictions, run_cv_evaluation
from forecasting.models import LSTMWrapper, NeuralNetWrapper


def load(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    data_start = "2023-01-16"
    data_end = "2025-01-05"
    # loading labels and features seperately due to previously saving an error, disregard
    publisher = PersistentParallelPublisher(pub_id, pub_ticker)

    labels = publisher.labels
    weights = publisher.label_weights
    features = publisher.features
    # lag features
    lagged_features = create_lagged_features(
        features,
        lag_periods=[1, 2],
        ma_periods=[4, 8],
        ema_periods=[4, 8],
    )
    # no TI features
    features = lagged_features.copy()
    if TI_features:
        # technical indicator features
        TI_features = create_TI_features(publisher.OHLC_data, 2, 4, 8)
        features = merge_features(lagged_features, TI_features)
    # for training and prediction: X - feature matrix, y - labels, w - weights
    X, y, w, feature_names, dates = prepare_ml_arrays(
        features, labels, weights, data_start, data_end
    )
    # align barrier dates
    publisher.take_profit_barriers = publisher.take_profit_barriers[dates]
    publisher.stop_loss_barriers = publisher.stop_loss_barriers[dates]

    run_feature_diagnostics(X, feature_names)
    return X, y, w, feature_names, dates


@handle("Random_Forest")
def random_forest_classifier(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """
    Train and evaluate Random Forest classifier on raw features.
    """
    print("\nRunning Random Forest Classifier on Raw Features...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    # hyperparam search over max_depth
    max_depths = [2, 3, 4, 5, 7, 10, 15, 20, 25, 40, 50, None]
    train_errs = []
    val_errs = []
    overall_results = []

    for depth in max_depths:

        def create_rf():
            return RandomForestClassifier(
                n_estimators=500, max_depth=depth, n_jobs=-1, random_state=11
            )

        results = run_cv_evaluation(
            model_fn=create_rf,
            X=X,
            y=y,
            dates=dates,
            weights=w,
            verbose=True,
            standardize=True,
        )

        train_errs.append(results["mean"]["training_loss"])
        val_errs.append(results["mean"]["validation_loss"])
        overall_results.append(results)

    # Plot error curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(max_depths)), train_errs, "b-", label="Training Error")
    plt.plot(range(len(max_depths)), val_errs, "r-", label="Validation Error")
    plt.xlabel("Max Depth Index")
    plt.ylabel("Error")
    plt.title("Random Forest Error vs Max Depth")
    plt.xticks(range(len(max_depths)), [str(d) if d else "None" for d in max_depths])
    plt.legend()
    # savefig(
    #     f"rf_error_curves{"_TI_features" if TI_features else ""}.png",
    #     path=f"{pub_id}/random_forest",
    # )
    min_index = np.argmin(val_errs)
    return overall_results[min_index]


@handle("Logistic")
def logistic_classifier(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """
    Train and evaluate Logistic Classification on raw features.
    """
    print("\nRunning Logistic Classification on Raw Features...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    # hyperparameter search over alpha
    alphas = 10.0 ** np.array([-4, -3, -2, -1, 0, 1, 2])

    train_errs = []
    val_errs = []
    overall_results = []

    for alpha in alphas:

        def create_logistic():
            return SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=alpha,
                max_iter=5000,
                tol=1e-3,
                random_state=11,
                shuffle=False,
            )

        results = run_cv_evaluation(
            model_fn=create_logistic,
            X=X,
            y=y,
            dates=dates,
            weights=w,
            verbose=True,
            standardize=True,
        )

        train_errs.append(results["mean"]["training_loss"])
        val_errs.append(results["mean"]["validation_loss"])
        overall_results.append(results)

    # Plot error curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(alphas)), train_errs, "b-", label="Training Error")
    plt.plot(range(len(alphas)), val_errs, "r-", label="Validation Error")
    plt.xlabel("Alpha")
    plt.ylabel("Error")
    plt.title("Logistic Classification Error vs Regularization Strength")
    plt.xticks(range(len(alphas)), [f"{alpha:.4f}" for alpha in alphas])
    plt.legend()
    # savefig(
    #     f"logistic_error_curves{"_TI_features" if TI_features else ""}.png",
    #     path=f"{pub_id}/logistic",
    # )
    min_index = np.argmin(val_errs)
    return overall_results[min_index]


@handle("RBF_Logistic")
def rbf_logistic_classifier(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """
    Train and evaluate RBF Kernel Logistic Classification on raw features.
    """
    print("\nRunning RBF Kernel Logistic Classification on Raw Features...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    sigma = 10.0
    alpha = 10.0

    # train_errs = np.full((len(sigmas), len(alphas)), 100.0)
    # val_errs = np.full((len(sigmas), len(alphas)), 100.0)
    # overall_results = np.full((len(sigmas), len(alphas)), 100.0)

    # for i, sigma in enumerate(sigmas):
    # Initialize RBF feature transform but don't make gramm matrix till cross-val
    rbf = GaussianRBFKernel(sigma)
    # for j, alpha in enumerate(alphas):

    def create_rbf_logistic():
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=5000,
            tol=1e-3,
            random_state=11,
            shuffle=False,
        )

    results = run_cv_evaluation(
        model_fn=create_rbf_logistic,
        X=X,
        y=y,
        dates=dates,
        weights=w,
        kernel=rbf,
        verbose=True,
        standardize=True,
    )

    # train_errs[i, j] = results["mean"]["training_loss"]
    # val_errs[i, j] = results["mean"]["validation_loss"]
    # overall_results[i, j] = results["mean"]

    # Plot error grids
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))

    # for (name, errs), ax in zip(
    #     [("training", train_errs), ("validation", val_errs)], axes
    # ):
    #     cax = ax.matshow(errs, norm=norm)
    #     ax.set_title(f"{name} errors")
    #     ax.set_ylabel(r"$\sigma$")
    #     ax.set_yticks(range(len(sigmas)))
    #     ax.set_yticklabels([f"{sigma:.4f}" for sigma in sigmas])
    #     ax.set_xlabel(r"$\alpha$")
    #     ax.set_xticks(range(len(alphas)))
    #     ax.set_xticklabels([f"{alpha:.4f}" for alpha in alphas], rotation=45)

    # fig.colorbar(cax)
    # savefig(
    #     f"rbf_error_grids{"_TI_features" if TI_features else ""}.png",
    #     f"{pub_id}/logistic",
    #     fig,
    # )
    # min_index = np.unravel_index(np.argmin(val_errs), val_errs.shape)
    return results


@handle("RBF_Logistic_PCA")
def rbf_logistic_pca(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """
    Train and evaluate RBF Kernel Logistic Classification on PCA-transformed features.
    """
    print("\nRunning RBF Kernel Logistic Classification on PCA Features...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    # Apply PCA
    pca = PCA()
    X_pca_full = pca.fit_transform(X)
    explained_var_ratio = pca.explained_variance_ratio_
    cumsum = np.cumsum(explained_var_ratio)

    # Find number of components for 95% variance
    n_components = np.argmax(cumsum >= 0.95) + 1
    print(f"\nUsing {n_components} PCA components explaining 95% of variance")

    # Apply PCA with optimal components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Grid search over sigma and alpha
    sigma = 10.0
    alpha = 10.0

    # train_errs = np.full((len(sigmas), len(alphas)), 100.0)
    # val_errs = np.full((len(sigmas), len(alphas)), 100.0)
    # overall_results = np.full((len(sigmas), len(alphas)), 100.0)

    # for i, sigma in enumerate(sigmas):
    # Initialize RBF feature transform but don't make gramm matrix till cross-val
    rbf = GaussianRBFKernel(sigma)
    # for j, alpha in enumerate(alphas):

    def create_rbf_logistic():
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=alpha,
            max_iter=5000,
            tol=1e-3,
            random_state=11,
            shuffle=False,
        )

    results = run_cv_evaluation(
        model_fn=create_rbf_logistic,
        X=X_pca,
        y=y,
        dates=dates,
        weights=w,
        kernel=rbf,
    )

    # train_errs[i, j] = results["mean"]["training_loss"]
    # val_errs[i, j] = results["mean"]["validation_loss"]
    # overall_results[i, j] = results["mean"]

    # Plot error grids
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    # norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))

    # for (name, errs), ax in zip(
    #     [("training", train_errs), ("validation", val_errs)], axes
    # ):
    #     cax = ax.matshow(errs, norm=norm)
    #     ax.set_title(f"{name} errors")
    #     ax.set_ylabel(r"$\sigma$")
    #     ax.set_yticks(range(len(sigmas)))
    #     ax.set_yticklabels([f"{sigma:.4f}" for sigma in sigmas])
    #     ax.set_xlabel(r"$\alpha$")
    #     ax.set_xticks(range(len(alphas)))
    #     ax.set_xticklabels([f"{alpha:.4f}" for alpha in alphas], rotation=45)

    # fig.colorbar(cax)
    # savefig(
    #     f"rbf_PCA_error_grids{"_TI_features" if TI_features else ""}.png",
    #     f"{pub_id}/logistic",
    #     fig,
    # )
    # min_index = np.unravel_index(np.argmin(val_errs), val_errs.shape)
    return results


@handle("Neural_Network")
def neural_network(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """Train and evaluate Neural Network classifier with hyperparameter optimization"""
    print("\nRunning Neural Network Classifier...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    # Grid search parameters
    hidden_size = (64, 64, 64)
    learning_rate = 1
    dropout = 0.3
    # train_errs = np.full((len(hidden_sizes), len(learning_rates)), 100.0)
    # val_errs = np.full((len(hidden_sizes), len(learning_rates)), 100.0)
    # overall_results = np.full((len(hidden_sizes), len(learning_rates)), 100.0)

    # for i, (hidden1, hidden2, hidden3) in enumerate(hidden_sizes):
    #     for j, learning_rate in enumerate(learning_rates):
    # Define network architecture
    # if hidden3:
    model = nn.Sequential(
        nn.Linear(X.shape[1], hidden_size[0]),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size[0], hidden_size[1]),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size[1], hidden_size[2]),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size[2], 1),
    )
    # else:
    # model = nn.Sequential(
    #     nn.Linear(X.shape[1], hidden1),
    #     nn.ReLU(),
    #     nn.Dropout(dropout),
    #     nn.Linear(hidden1, hidden2),
    #     nn.ReLU(),
    #     nn.Dropout(dropout),
    #     nn.Linear(hidden2, 1),
    # )

    def create_nn():
        return NeuralNetWrapper(
            model=model,
            optimizer_class=optim.Adam,
            learning_rate=learning_rate,
            epochs=5000,
            batch_size=32,
        )

    results = run_cv_evaluation(
        model_fn=create_nn,
        X=X,
        y=y,
        dates=dates,
        weights=w,
        verbose=True,
        standardize=True,
    )

    # train_errs[i, j] = results["mean"]["training_loss"]
    # val_errs[i, j] = results["mean"]["validation_loss"]
    # overall_results[i, j] = results["mean"]

    # Also use single train-validation split so we can compare performance to LSTM
    split_idx = int(len(X) * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, len(X))

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X[train_idx])
    X_val_standardized, *_ = standardize_cols(X[val_idx], mu, sigma)
    # Create model and train
    model = create_nn()
    model.fit(X_train_standardized, y[train_idx], sample_weight=w[train_idx])

    # Generate predictions
    train_pred = model.predict(X_train_standardized)
    val_pred = model.predict(X_val_standardized)

    last_fold_results = evaluate_predictions(
        y[train_idx], train_pred, w[train_idx], y[val_idx], val_pred, w[val_idx]
    )

    # Plot error grids
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    # norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))

    # for (name, errs), ax in zip(
    #     [("training", train_errs), ("validation", val_errs)], axes
    # ):
    #     cax = ax.matshow(errs, norm=norm)
    #     ax.set_title(f"{name} errors")
    #     ax.set_ylabel("Hidden Layer Sizes")
    #     ax.set_yticks(range(len(hidden_sizes)))
    #     ax.set_yticklabels(
    #         [f"({h1}, {h2}{", "+str(h3) if h3 else ""})" for h1, h2, h3 in hidden_sizes]
    #     )
    #     ax.set_xlabel("Learning Rate")
    #     ax.set_xticks(range(len(learning_rates)))
    #     ax.set_xticklabels([f"{lr:.4f}" for lr in learning_rates], rotation=45)

    # fig.colorbar(cax)
    # savefig(
    #     f"nn_error_grids_learning_rate{"_TI_features" if TI_features else ""}.png",
    #     f"{pub_id}/neural_net",
    #     fig,
    # )

    # min_index = np.unravel_index(np.argmin(val_errs), val_errs.shape)
    print(results)
    print(last_fold_results)
    return results, last_fold_results


@handle("LSTM")
def lstm_classifier(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    """
    Train and evaluate LSTM classifier with hyperparameter optimization
    """
    print("\nRunning LSTM Classifier...")
    X, y, w, feature_names, dates = load(pub_id, pub_ticker, TI_features)

    # Grid search parameters
    hidden_dim = 64
    sequence_length = 12
    base_lr = 1
    # Initial learning rates for inverse square root schedule
    hidden_layer_depth = 3
    dropout = 0.3

    # train_errs = np.full((len(hidden_dims), len(sequence_lengths)), 100.0)
    # val_errs = np.full((len(hidden_dims), len(sequence_lengths)), 100.0)
    # overall_results = np.full((len(hidden_dims), len(sequence_lengths)), 100.0)

    # for i, hidden_dim in enumerate(hidden_dims):
    #     for d, seq_len in enumerate(sequence_lengths):

    def create_lstm():
        return LSTMWrapper(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=hidden_layer_depth,
            sequence_length=sequence_length,
            base_lr=base_lr,
            min_lr=1e-5,
            epochs=5000,
            batch_size=32,
            early_stopping_patience=10,
        )

    # Use single train-validation split for LSTM
    split_idx = int(len(X) * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, len(X))

    # Standardize features
    X_train_standardized, mu, sigma = standardize_cols(X[train_idx])
    X_val_standardized, *_ = standardize_cols(X[val_idx], mu, sigma)
    # Create model and train
    model = create_lstm()
    model.fit(X_train_standardized, y[train_idx], sample_weight=w[train_idx])

    # Generate predictions
    train_pred = model.predict(X_train_standardized)
    val_pred = model.predict(X_val_standardized)

    results = evaluate_predictions(
        y[train_idx], train_pred, w[train_idx], y[val_idx], val_pred, w[val_idx]
    )
    # train_err = results["training_loss"]
    # val_err = results["validation_loss"]
    # train_errs[i, d] = train_err
    # val_errs[i, d] = val_err
    # overall_results[i, d] = results

    # Plot error grids
    # fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    # norm = plt.Normalize(vmin=0, vmax=max(train_errs.max(), val_errs.max()))

    # for (name, errs), ax in zip(
    #     [("training", train_errs), ("validation", val_errs)], axes
    # ):
    #     cax = ax.matshow(errs, norm=norm)
    #     ax.set_title(f"{name} errors")
    #     ax.set_ylabel("Hidden Layer Sizes")
    #     ax.set_yticks(range(len(hidden_dims)))
    #     ax.set_yticklabels([f"{dim}" for dim in hidden_dims])
    #     ax.set_xlabel("Sequence Length")
    #     ax.set_xticks(range(len(sequence_lengths)))
    #     ax.set_xticklabels([f"{lr}" for lr in sequence_lengths], rotation=45)

    # fig.colorbar(cax)
    # savefig(
    #     f"lstm_error_grids_sequence_length{"_TI_features" if TI_features else ""}.png",
    #     f"{pub_id}/lstm",
    #     fig,
    # )
    # min_index = np.unravel_index(np.argmin(val_errs), val_errs.shape)
    return results


@handle("doItAll")
def doItAll():
    # Publishers
    PUBLISHERS = {
        "SEGA": "6460.T",
        "Electronic Arts": "EA",
        "Ubisoft": "UBI.PA",
        "Square Enix": "9684.T",
        "Frontier Developments": "FDEV.L",
        "CD PROJEKT RED": "CDR.WA",
    }

    for pub_id, pub_ticker in PUBLISHERS.items():
        report = ReportFormatter()

        # Random Forest
        RF_Results = random_forest_classifier(pub_id, pub_ticker, True)
        RF_Results_TI_features = random_forest_classifier(pub_id, pub_ticker, True)
        report.add_result(
            pub_id,
            "Random Forest",
            results=RF_Results,
            results_TI_features=RF_Results_TI_features,
        )
        # Logistic Classification
        Logistic_Results = logistic_classifier(pub_id, pub_ticker)
        Logistic_Results_TI_features = logistic_classifier(pub_id, pub_ticker, True)
        report.add_result(
            pub_id,
            "Logistic Classification",
            results=Logistic_Results,
            results_TI_features=Logistic_Results_TI_features,
        )

        # Logistic Classification with RBF basis
        Logistic_Results = rbf_logistic_classifier(pub_id, pub_ticker)
        Logistic_Results_TI_features = rbf_logistic_classifier(pub_id, pub_ticker, True)
        report.add_result(
            pub_id,
            "Logistic Classification with RBF basis",
            results=Logistic_Results,
            results_TI_features=Logistic_Results_TI_features,
        )

        # Logistic Classification with RBF basis and PCA Features
        Logistic_Results = rbf_logistic_pca(pub_id, pub_ticker)
        Logistic_Results_TI_features = rbf_logistic_pca(pub_id, pub_ticker, True)
        report.add_result(
            pub_id,
            "Logistic Classification with RBF basis and PCA Features",
            results=Logistic_Results,
            results_TI_features=Logistic_Results_TI_features,
        )

        # Neural Network
        NN_Results, NN_Results_Last_Fold = neural_network(pub_id, pub_ticker)
        NN_Results_TI_features, NN_Results_TI_features_Last_Fold = neural_network(
            pub_id, pub_ticker, True
        )
        report.add_result(
            pub_id,
            "Neural Network",
            results=NN_Results,
            results_TI_features=NN_Results_TI_features,
        )
        report.add_result(
            pub_id,
            "Neural Network without purged CV",
            results=NN_Results_Last_Fold,
            results_TI_features=NN_Results_TI_features_Last_Fold,
        )
        # LSTM
        LSTM_Results = lstm_classifier(pub_id, pub_ticker)
        LSTM_Results_TI_features = lstm_classifier(pub_id, pub_ticker, True)
        report.add_result(
            pub_id,
            "LSTM",
            results=LSTM_Results,
            results_TI_features=LSTM_Results_TI_features,
        )

        # Save final report
        report.save_report(f"figs/{pub_id}/results.txt")

    return None


def evaluate_test_set(
    model_fn, X_train, y_train, w_train, X_test, y_test, w_test, standardize=False
):
    """
    Evaluate model performance on held-out test set.

    Args:
        model_fn: Function that returns a fresh model instance
        X_train: Training feature matrix
        y_train: Training labels
        w_train: Training sample weights
        X_test: Test feature matrix
        y_test: Test labels
        w_test: Test sample weights
        standardize: Whether to standardize features

    Returns:
        dict: Dictionary containing test set performance metrics
    """
    if standardize:
        # Standardize features using only training data statistics
        X_train_std, mu, sigma = standardize_cols(X_train)
        X_test_std, *_ = standardize_cols(X_test, mu, sigma)
    else:
        X_train_std, X_test_std = X_train, X_test
    # Initialize and train model
    model = model_fn()
    model.fit(X_train_std, y_train, sample_weight=w_train)

    # Generate predictions
    y_train_pred = model.predict(X_train_std)
    y_test_pred = model.predict(X_test_std)

    # Evaluate performance
    return evaluate_predictions(
        y_train, y_train_pred, w_train, y_test, y_test_pred, w_test
    )


def run_test_evaluation(
    model_fn,
    X,
    y,
    w,
    dates,
    train_end_date="2024-09-08",
    kernel=None,
    standardize=False,
):
    """
    Run evaluation on chronological train/test split.

    Args:
        model_fn: Function that returns a fresh model instance
        X: Feature matrix
        y: Labels
        w: Sample weights
        dates: Datetime index
        train_end_date: Cut-off date for training data
        kernel: Optional kernel function
        standardize: Whether to standardize features

    Returns:
        dict: Dictionary containing test set performance metrics
    """
    # Create train/test split
    train_mask = dates <= train_end_date
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    w_train, w_test = w[train_mask], w[~train_mask]

    if kernel:
        X_test = kernel(X_test, X)
        X_train = kernel(X_train, X)

    return evaluate_test_set(
        model_fn, X_train, y_train, w_train, X_test, y_test, w_test, standardize
    )


@handle("test_evaluation")
def run_all_test_evaluations():
    """Run test set evaluation for all models across all publishers."""

    PUBLISHERS = {
        "SEGA": "6460.T",
        "Electronic Arts": "EA",
        "Ubisoft": "UBI.PA",
        "Square Enix": "9684.T",
        "Frontier Developments": "FDEV.L",
        "CD PROJEKT RED": "CDR.WA",
    }

    for pub_id, pub_ticker in PUBLISHERS.items():
        report = ReportFormatter()

        # Load data with and without TI features
        X, y, w, feature_names, dates = load(pub_id, pub_ticker, False)
        X_TI, y_TI, w_TI, feature_names_TI, dates_TI = load(pub_id, pub_ticker, True)

        # Random Forest
        def create_rf():
            return RandomForestClassifier(
                n_estimators=500, max_depth=5, n_jobs=-1, random_state=11
            )

        RF_Results = run_test_evaluation(create_rf, X, y, w, dates, standardize=True)
        RF_Results_TI = run_test_evaluation(
            create_rf, X_TI, y_TI, w_TI, dates_TI, standardize=True
        )
        report.add_result(pub_id, "Random Forest", RF_Results, RF_Results_TI)

        # Logistic Classification
        def create_logistic():
            return SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1,
                max_iter=5000,
                tol=1e-3,
                random_state=11,
                shuffle=False,
            )

        Log_Results = run_test_evaluation(
            create_logistic, X, y, w, dates, standardize=True
        )
        Log_Results_TI = run_test_evaluation(
            create_logistic, X_TI, y_TI, w_TI, dates_TI, standardize=True
        )
        report.add_result(
            pub_id, "Logistic Classification", Log_Results, Log_Results_TI
        )

        # RBF Kernel Logistic
        rbf = GaussianRBFKernel(1.0)

        def create_logistic_rbf():
            return SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=10,
                max_iter=5000,
                tol=1e-3,
                random_state=11,
                shuffle=False,
            )

        RBF_Results = run_test_evaluation(
            create_logistic_rbf, X, y, w, dates, kernel=rbf, standardize=True
        )
        RBF_Results_TI = run_test_evaluation(
            create_logistic_rbf,
            X_TI,
            y_TI,
            w_TI,
            dates_TI,
            kernel=rbf,
            standardize=True,
        )
        report.add_result(pub_id, "RBF Kernel Logistic", RBF_Results, RBF_Results_TI)

        # Neural Network
        def create_nn():
            model = nn.Sequential(
                nn.Linear(X.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            return NeuralNetWrapper(
                model=model,
                optimizer_class=optim.Adam,
                learning_rate=1,
                epochs=5000,
                batch_size=32,
            )

        def create_nn_ti():
            model = nn.Sequential(
                nn.Linear(X_TI.shape[1], 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )
            return NeuralNetWrapper(
                model=model,
                optimizer_class=optim.Adam,
                learning_rate=1,
                epochs=5000,
                batch_size=32,
            )

        NN_Results = run_test_evaluation(create_nn, X, y, w, dates, standardize=True)
        NN_Results_TI = run_test_evaluation(
            create_nn_ti, X_TI, y_TI, w_TI, dates_TI, standardize=True
        )
        report.add_result(pub_id, "Neural Network", NN_Results, NN_Results_TI)

        # LSTM
        def create_lstm():
            return LSTMWrapper(
                hidden_dim=64,
                dropout=0.25,
                num_layers=3,
                sequence_length=12,
                base_lr=1,
                min_lr=1e-5,
                epochs=5000,
                batch_size=32,
                early_stopping_patience=10,
            )

        LSTM_Results = run_test_evaluation(
            create_lstm, X, y, w, dates, standardize=True
        )
        LSTM_Results_TI = run_test_evaluation(
            create_lstm, X_TI, y_TI, w_TI, dates_TI, standardize=True
        )
        report.add_result(pub_id, "LSTM", LSTM_Results, LSTM_Results_TI)

        # Save test results
        report.save_report(f"results/{pub_id}/testSet_results.txt")

    return None


@handle("load")
def doItAll():
    # Publishers
    PUBLISHERS = {
        "SEGA": "6460.T",
        "Electronic Arts": "EA",
        "Ubisoft": "UBI.PA",
        "Square Enix": "9684.T",
        "Frontier Developments": "FDEV.L",
        "CD PROJEKT RED": "CDR.WA",
    }

    for pub_id, pub_ticker in PUBLISHERS.items():
        PersistentParallelPublisher(pub_id, pub_ticker)


@handle("BlackLitterman")
def run_black_litterman_evaluation():
    """
    Implements Black-Litterman portfolio optimization using neural network predictions.

    Key components:
    1. Loads predictions from best performing model (Neural Network)
    2. Constructs views and confidence levels from predictions
    3. Optimizes portfolio weights using Black-Litterman model
    4. Evaluates portfolio performance over test period
    """
    PUBLISHERS = {
        "SEGA": "6460.T",
        "Electronic Arts": "EA",
        "Ubisoft": "UBI.PA",
        "Square Enix": "9684.T",
        "Frontier Developments": "FDEV.L",
        "CD PROJEKT RED": "CDR.WA",
    }

    # Initialize publishers with data
    publisher_data = {}
    for pub_id, pub_ticker in PUBLISHERS.items():
        publisher = PersistentParallelPublisher(pub_id, pub_ticker)
        publisher_data[pub_id] = publisher

    # Generate Neural Network predictions for test period
    predictions = {}
    probabilities = {}
    barriers = {}

    for pub_id, publisher in publisher_data.items():
        # Load data with features
        X, y, w, feature_names, dates = load(pub_id, PUBLISHERS[pub_id], False)
        print(f"dates: {dates}")
        dateIndex = (
            pd.DatetimeIndex(dates).tz_convert("UTC").tz_localize(None).normalize()
        )
        print(f"dates after trans: {dateIndex}")
        # Create train/test split using chronological dates
        train_end_date = pd.Timestamp("2024-09-08")
        train_mask = dateIndex <= train_end_date
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        w_train, w_test = w[train_mask], w[~train_mask]
        test_dates = dateIndex[~train_mask]

        # Initialize and train neural network
        model = nn.Sequential(
            nn.Linear(X.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
        nn_model = NeuralNetWrapper(
            model=model,
            optimizer_class=optim.Adam,
            learning_rate=0.0001,
            epochs=5000,
            batch_size=32,
        )

        # Standardize features
        X_train_std, mu, sigma = standardize_cols(X_train)
        X_test_std, *_ = standardize_cols(X_test, mu, sigma)

        # Train model and generate predictions
        nn_model.fit(X_train_std, y_train, sample_weight=w_train)
        print(f"test dates: {test_dates}")
        predictions[pub_id] = pd.Series(nn_model.predict(X_test_std), index=test_dates)
        probabilities[pub_id] = pd.Series(
            nn_model.predict_proba(X_test_std), index=test_dates
        )

        # Store barriers
        barriers[pub_id] = pd.DataFrame(
            {
                "take_profit": publisher.take_profit_barriers[dates][~train_mask],
                "stop_loss": publisher.stop_loss_barriers[dates][~train_mask],
            }
        )

    # Convert predictions and probabilities to DataFrames
    predictions_df = pd.DataFrame(predictions)
    probabilities_df = pd.DataFrame(probabilities)

    # Combine barriers into single DataFrame with MultiIndex
    barriers_df = pd.concat(
        [df.assign(publisher=pub_id) for pub_id, df in barriers.items()]
    ).set_index("publisher", append=True)

    # Initialize and run Black-Litterman portfolio optimization
    portfolio = GamePublisherBLPortfolio(
        PUBLISHERS, start_date="2024-09-08", end_date="2025-01-07"
    )

    bl_nn_metrics = portfolio.backtest_strategy(
        predictions_df, probabilities_df, barriers_df
    )
    bl_base_metrics = portfolio.backtest_baseline_bl_portfolio()
    reporter = PortfolioReportFormatter()
    # Save results
    reporter.add_result("BL with Neural Network Views", bl_nn_metrics)
    reporter.add_result("BL Market Equilibrium", bl_base_metrics)
    reporter.save_report("portfolio_results.txt")

    # Print results
    print("\nNN Portfolio Performance Metrics:")
    print(f"Cumulative Return: {bl_nn_metrics.cumulative_returns.iloc[-1]:.4f}")
    print(f"Sharpe Ratio: {bl_nn_metrics.sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {bl_nn_metrics.max_drawdown:.4f}")
    print("\nBase Portfolio Performance Metrics:")
    print(f"Cumulative Return: {bl_base_metrics.cumulative_returns.iloc[-1]:.4f}")
    print(f"Sharpe Ratio: {bl_base_metrics.sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {bl_base_metrics.max_drawdown:.4f}")

    # Plot cumulative returns
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)

    # Define color palette
    colors = {
        "BL-NN": "#2E86AB",  # Strong blue for our primary strategy
        "BL-Base": "#A23B72",  # Deep raspberry for baseline
    }

    # Plot each strategy with enhanced styling
    bl_nn_metrics.cumulative_returns.plot(
        label="Black-Litterman with Neural Network",
        color=colors["BL-NN"],
        alpha=0.9,
        linewidth=2,
        ax=ax,
    )

    bl_base_metrics.cumulative_returns.plot(
        label="Black-Litterman Baseline",
        color=colors["BL-Base"],
        alpha=0.9,
        linewidth=2,
        ax=ax,
    )

    # Enhance grid and spines
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Configure title and labels with professional formatting
    ax.set_title(
        "Cumulative Returns Comparison", fontsize=14, pad=20, fontweight="bold"
    )
    ax.set_xlabel("Date", fontsize=12, labelpad=10)
    ax.set_ylabel("Cumulative Return", fontsize=12, labelpad=10)

    # Configure legend with optimal placement and styling
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10,
    )

    # Adjust layout to prevent cutoff
    plt.tight_layout()

    # Save with high DPI for quality
    plt.savefig(
        "strategy_comparison.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close()

    return bl_nn_metrics


if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    main()
