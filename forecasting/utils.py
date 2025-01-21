import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ta import momentum, trend, volatility


def align_weekly_dates(features_df, labels_df, weights_df):
    # align dates between features DataFrame (with PeriodIndex) and labels/weights Series (with market open times).

    # create a mapping from period to market open time
    week_to_market = {}

    # convert PeriodIndex start dates to datetime
    for period in features_df.index:
        # find the corresponding market open time
        for market_time in labels_df.index:
            market_date = pd.to_datetime(market_time).date()
            if market_date == period.date() + pd.Timedelta(days=1):
                week_to_market[period] = market_time
                break
    # create new features DataFrame with market open time index
    aligned_features = features_df.copy()
    aligned_features.index = [week_to_market.get(idx) for idx in aligned_features.index]
    aligned_features = aligned_features.dropna(
        axis=0, how="all"
    )  # drop rows where we couldn't find a matching market time

    # now all DataFrames have market open time index
    common_dates = aligned_features.index.intersection(labels_df.index)

    return (
        aligned_features.loc[common_dates],
        labels_df.loc[common_dates],
        weights_df.loc[common_dates],
    )


def create_lagged_features(
    features_df, lag_periods=[1, 2], ma_periods=[4, 8], ema_periods=[3, 6]
):
    # create lagged features with proper handling of index alignment

    # convert PeriodIndex to DatetimeIndex using the start of each period
    features_df = features_df.copy()

    features_df.index = [period.end_time for period in features_df.index]

    # start with the original features
    lagged_features = [features_df.copy()]

    # get list of original feature names
    feature_names = features_df.columns

    for feature in feature_names:
        feature_series = features_df[feature]
        feature_transformations = []

        # simple lags
        for lag in lag_periods:
            lagged = feature_series.shift(lag)
            lagged_df = pd.Series(
                data=lagged, index=features_df.index, name=f"{feature}_lag_{lag}"
            )
            feature_transformations.append(lagged_df)

        # moving averages
        for ma in ma_periods:
            ma_series = feature_series.rolling(window=ma, min_periods=1).mean()
            ma_df = pd.Series(
                data=ma_series, index=features_df.index, name=f"{feature}_ma_{ma}"
            )
            feature_transformations.append(ma_df)

        # exponential moving averages
        for ema in ema_periods:
            ema_series = feature_series.ewm(
                span=ema, adjust=False, min_periods=1
            ).mean()
            ema_df = pd.Series(
                data=ema_series, index=features_df.index, name=f"{feature}_ema_{ema}"
            )
            feature_transformations.append(ema_df)

        lagged_features.extend(feature_transformations)
    # concatenate all features
    result = pd.concat(lagged_features, axis=1)

    # forward fill to handle NaN values from lagging
    result = result.ffill()
    # fill any remaining NaNs with 0
    result = result.fillna(0)

    return result


def create_TI_features(OHLC, short_period, medium_period, long_period):
    # Convert weeks to hours for our lookback periods
    short_period = short_period * 5 * 6  # weeks * 5 trading days * 6 trading hours
    medium_period = medium_period * 5 * 6
    long_period = long_period * 5 * 6

    TI_features = OHLC.copy()
    TI_features.index = TI_features.index.tz_convert("UTC").tz_localize(None)
    # Calculate SMAs
    for period in [short_period, medium_period, long_period]:
        TI_features[f"SMA_{period}"] = trend.sma_indicator(
            close=TI_features["Close"].squeeze(), window=period, fillna=True
        )

    # Calculate EMAs
    for period in [short_period, medium_period, long_period]:
        TI_features[f"EMA_{period}"] = trend.ema_indicator(
            close=TI_features["Close"].squeeze(), window=period, fillna=True
        )

    # Calculate ATR
    for period in [short_period, medium_period, long_period]:
        TI_features[f"ATR_{period}"] = volatility.average_true_range(
            high=TI_features["High"].squeeze(),
            low=TI_features["Low"].squeeze(),
            close=TI_features["Close"].squeeze(),
            window=period,
            fillna=True,
        )

    # Calculate RSI
    for period in [short_period, medium_period, long_period]:
        TI_features[f"RSI_{period}"] = momentum.rsi(
            close=TI_features["Close"].squeeze(), window=period, fillna=True
        )

    weekly_TI_features = TI_features.groupby(pd.Grouper(freq="W-SUN")).agg(
        {col: "last" for col in TI_features.columns}
    )
    weekly_TI_features.columns = weekly_TI_features.columns.droplevel(1)

    return weekly_TI_features


def merge_features(lagged_features, TI_features):
    lagged_features.index = pd.to_datetime(lagged_features.index.date)
    TI_features.index = pd.to_datetime(TI_features.index.date)
    print(TI_features)
    print(lagged_features)

    features = pd.merge(
        lagged_features,
        TI_features,
        left_index=True,
        right_index=True,
        how="left",
    )
    print(features)
    return features


def prepare_ml_arrays(features_df, labels_df, weights_df, start_date, end_date):
    # prepare data for ML by aligning label vector and feature matrix dates

    # convert start_date and end_date to timezone-aware timestamps matching labels
    if not isinstance(start_date, pd.Timestamp):
        # get timezone from labels index
        tz = labels_df.index[0].tz
        start_date = pd.to_datetime(start_date).tz_localize(tz)
        end_date = pd.to_datetime(end_date).tz_localize(tz)

    # convert labels and weights to DataFrame if they're Series
    if isinstance(labels_df, pd.Series):
        labels_df = labels_df.to_frame()
    if isinstance(weights_df, pd.Series):
        weights_df = weights_df.to_frame()

    # align dates between features and labels/weights
    features_aligned, labels_aligned, weights_aligned = align_weekly_dates(
        features_df, labels_df, weights_df
    )
    # filter to specified date range
    mask = (features_aligned.index >= start_date) & (features_aligned.index <= end_date)
    features_aligned = features_aligned[mask]
    labels_aligned = labels_aligned[mask]
    weights_aligned = weights_aligned[mask]

    # extract arrays
    X = features_aligned.values
    y = labels_aligned[labels_aligned.columns[0]].values
    y[y == 0] = -1
    w = weights_aligned[weights_aligned.columns[0]].values

    # have sample weights from relative distances to barriers
    # add 1 to all labels due to potential 0 returns
    w += 1

    return X, y, w, list(features_aligned.columns), features_aligned.index


def standardize_cols(X, mu=None, sigma=None):
    # standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.0

    return (X - mu) / sigma, mu, sigma


def generate_feature_diagnostics(X, feature_names):
    # generate comprehensive diagnostics for feature matrix.

    X_df = pd.DataFrame(X, columns=feature_names)

    diagnostics = {
        "shape": X.shape,
        "total_elements": X.size,
        "counts": {
            "inf": np.sum(np.isinf(X)),
            "-inf": np.sum(np.isneginf(X)),
            "nan": np.sum(np.isnan(X)),
            "none": np.sum(X == None),
            "zero": np.sum(X == 0),
        },
        "problematic_features": {
            "inf_features": [],
            "nan_features": [],
            "zero_features": [],
            "constant_features": [],
        },
        "value_ranges": {},
    }

    # analyze each feature
    for feature in feature_names:
        feature_values = X_df[feature]

        # check for infinities
        if np.any(np.isinf(feature_values)):
            diagnostics["problematic_features"]["inf_features"].append(feature)

        # Check for NaNs
        if feature_values.isna().any():
            diagnostics["problematic_features"]["nan_features"].append(feature)

        # check for all zeros
        if (feature_values == 0).all():
            diagnostics["problematic_features"]["zero_features"].append(feature)

        # check for constant values
        if feature_values.nunique() == 1:
            diagnostics["problematic_features"]["constant_features"].append(feature)

        # get value range
        diagnostics["value_ranges"][feature] = {
            "min": float(feature_values.min()),
            "max": float(feature_values.max()),
            "mean": float(feature_values.mean()),
            "std": float(feature_values.std()),
        }

    return diagnostics


def print_feature_diagnostics(diagnostics):
    # print formatted diagnostic report.

    print("\n=== FEATURE MATRIX DIAGNOSTIC REPORT ===")
    print(f"\nShape: {diagnostics['shape']}")
    print(f"Total Elements: {diagnostics['total_elements']}")

    print("\nValue Counts:")
    print(f"Infinity: {diagnostics['counts']['inf']}")
    print(f"Negative Infinity: {diagnostics['counts']['-inf']}")
    print(f"NaN: {diagnostics['counts']['nan']}")
    print(f"None: {diagnostics['counts']['none']}")
    print(f"Zero: {diagnostics['counts']['zero']}")

    print("\nProblematic Features:")
    print(
        f"Features with infinities: {len(diagnostics['problematic_features']['inf_features'])}"
    )
    print(
        f"Features with NaNs: {len(diagnostics['problematic_features']['nan_features'])}"
    )
    print(
        f"Features that are all zero: {len(diagnostics['problematic_features']['zero_features'])}"
    )
    print(
        f"Constant features: {len(diagnostics['problematic_features']['constant_features'])}"
    )

    # print details of problematic features if they exist
    if diagnostics["problematic_features"]["zero_features"]:
        print("\nFeatures that are all zero:")
        for feature in diagnostics["problematic_features"]["zero_features"]:
            print(f"- {feature}")

    if diagnostics["problematic_features"]["constant_features"]:
        print("\nConstant features:")
        for feature in diagnostics["problematic_features"]["constant_features"]:
            value_range = diagnostics["value_ranges"][feature]
            print(f"- {feature} (constant value: {value_range['mean']})")

    # print summary statistics for a few problematic features (if any exist)
    problematic = (
        diagnostics["problematic_features"]["inf_features"]
        + diagnostics["problematic_features"]["nan_features"]
    )
    if problematic:
        print("\nSummary statistics for some problematic features:")
        for feature in problematic:
            value_range = diagnostics["value_ranges"][feature]
            print(f"\n{feature}:")
            print(f"  Range: [{value_range['min']}, {value_range['max']}]")
            print(f"  Mean: {value_range['mean']:.4f}")
            print(f"  Std: {value_range['std']:.4f}")


def run_feature_diagnostics(X, feature_names):
    # run and print feature diagnostics.

    diagnostics = generate_feature_diagnostics(X, feature_names)
    print_feature_diagnostics(diagnostics)
    return diagnostics


class ReportFormatter:
    """Simple tracker for performance across companies."""

    def __init__(self):
        self.results_dict = {}

    def add_result(
        self,
        company,
        model,
        results,
        results_TI_features,
    ):
        """Record model performance for a company."""
        if company not in self.results_dict:
            self.results_dict[company] = {}

        if model not in self.results_dict[company]:
            self.results_dict[company][model] = {
                "raw_features": results,
                "TI_features": results_TI_features,
            }

    def save_report(self, filename: str):
        """Save performance summary to a clean, readable file."""
        with open(filename, "w") as f:
            for company in self.results_dict:
                f.write(f"\nCompany: {company}\n")
                f.write("-" * 40 + "\n")

                for model in self.results_dict[company]:
                    try:
                        model_results = self.results_dict[company][model]
                        raw_feature_results = model_results["raw_features"]
                        metrics = raw_feature_results["mean"]
                        f.write(f"\nModel: {model}")
                        for metric in metrics:
                            formatted_metric = metric.replace("_", " ").title()
                            f.write(
                                f"\n{formatted_metric}: {raw_feature_results['mean'][metric]:.4f} +/- {raw_feature_results['std'][metric]:.4f}"
                            )
                        f.flush()
                        TI_feature_results = model_results["TI_features"]
                        f.write(f"\nModel: {model} with Technical Indicator Features")
                        for metric in TI_feature_results["mean"]:
                            formatted_metric = metric.replace("_", " ").title()
                            f.write(
                                f"\n{formatted_metric}: {raw_feature_results['mean'][metric]:.4f} +/- {raw_feature_results['std'][metric]:.4f}"
                            )
                    except:
                        model_results = self.results_dict[company][model]
                        f.write(f"\nModel: {model}")
                        raw_feature_results = model_results["raw_features"]
                        f.write(f"\nAccuracy: {raw_feature_results['accuracy']:.4f}")
                        f.write(f"\nPrecision: {raw_feature_results['precision']:.4f}")
                        f.write(f"\nRecall: {raw_feature_results['recall']:.4f}")
                        f.write(f"\nF1 Score: {raw_feature_results['f1']:.4f}")
                        f.write(
                            f"\nValidation Loss: {raw_feature_results['validation_loss']:.4f}"
                        )
                        f.write(
                            f"\nTraining Loss: {raw_feature_results['training_loss']:.4f}"
                        )
                        f.write(f"\nModel: {model} with Technical Indicator Features")
                        TI_feature_results = model_results["TI_features"]
                        f.write(f"\nAccuracy: {TI_feature_results['accuracy']:.4f}")
                        f.write(f"\nPrecision: {TI_feature_results['precision']:.4f}")
                        f.write(f"\nRecall: {TI_feature_results['recall']:.4f}")
                        f.write(f"\nF1 Score: {TI_feature_results['f1']:.4f}")
                        f.write(
                            f"\nValidation Loss: {TI_feature_results['validation_loss']:.4f}"
                        )
                        f.write(
                            f"\nTraining Loss: {TI_feature_results['training_loss']:.4f}"
                        )


def savefig(fname, path="", fig=None, verbose=True):
    path = Path("", f"figs/{path}", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=0)
    if verbose:
        print(f"Figure saved as '{path}'")


################################################################################
# Helpers for setting up the command-line interface

_funcs = {}


def handle(name):
    def register(func):
        _funcs[name] = func
        return func

    return register


def run(model):
    if model not in _funcs:
        raise ValueError(f"unknown model {model}")
    return _funcs[model]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(_funcs.keys()) + ["all"])
    args = parser.parse_args()
    if args.model == "all":
        for q in sorted(_funcs.keys()):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.model)
