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
from publisher import *
from forecasting.purged_CV import evaluate_predictions, run_cv_evaluation
from forecasting.models import LSTMWrapper, NeuralNetWrapper


def load(pub_id="SEGA", pub_ticker="6460.T", TI_features=False):
    # loading labels and features seperately due to previously saving an error, disregard
    OHLC_pub = ParallelPublisher(pub_id, pub_ticker)
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
        TI_features = create_TI_features(OHLC_pub.OHLC_data, 2, 4, 8)
        features = merge_features(lagged_features, TI_features)
    # for training and prediction: X - feature matrix, y - labels, w - weights
    X, y, w, feature_names, dates = prepare_ml_arrays(
        features, labels, weights, "2023-01-01", "2024-09-08"
    )
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
    learning_rate = 0.1
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
            epochs=1000,
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


if __name__ == "__main__":
    pd.set_option("future.no_silent_downcasting", True)
    # pd.set_option("display.max_columns", None)
    # pd.set_option("display.max_rows", None)
    main()
