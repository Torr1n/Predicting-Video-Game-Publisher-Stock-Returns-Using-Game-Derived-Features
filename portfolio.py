import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import BlackLittermanModel, EfficientFrontier, risk_models
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics"""

    cumulative_returns: pd.Series
    sharpe_ratio: float
    max_drawdown: float


class GamePublisherBLPortfolio:
    def __init__(
        self,
        publishers: Dict[str, str],
        start_date: str,
        end_date: str,
        risk_free_rate: float = 0.045,  # 3 month treasury bill rate as of Sept 2024
    ):
        self.publishers = publishers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.risk_free_rate = risk_free_rate
        self.weekly_rf_rate = (1 + risk_free_rate) ** (1 / 52) - 1

        # Initialize market data
        self._initialize_market_data()

    def _initialize_market_data(self):
        """
        Initialize market data and parameters for Black-Litterman model

        Calculates:
        - Weekly returns for covariance estimation
        - Market capitalizations for equilibrium returns
        - Risk aversion parameter using market data

        Note: Risk aversion parameter is derived from market returns using
        the relationship λ = (E[r_m] - r_f) / σ²_m, where:
        - E[r_m] is the annualized market return
        - r_f is the risk-free rate
        - σ²_m is the annualized market variance
        """
        # Fetch price data
        tickers = list(self.publishers.values())
        prices = yf.download(
            tickers,
            start=self.start_date + pd.Timedelta(days=1) - pd.Timedelta(days=14),
            end=self.end_date,
        )["Adj Close"]

        # Calculate weekly returns
        self.returns = prices.resample("W-MON").last().pct_change()
        print(f"Returns {self.returns}")

        # Get market caps
        self.market_caps = {}
        for name, ticker in self.publishers.items():
            stock = yf.Ticker(ticker)
            self.market_caps[name] = stock.info["marketCap"]

        # Calculate covariance matrix
        self.cov_matrix = self.returns.cov() * 52  # Annualize

        # Calculate market-implied risk aversion
        market_prices = yf.download("SPY", start=self.start_date, end=self.end_date)[
            "Adj Close"
        ]
        market_returns = market_prices.resample("W").last().pct_change()

        # Extract scalar values for risk aversion calculation
        market_var = float(market_returns.var() * 52)  # Annualize variance
        market_mean = float(market_returns.mean() * 52)  # Annualize mean

        # Calculate risk aversion as scalar value
        self.risk_aversion = (market_mean - self.risk_free_rate) / market_var

    def backtest_baseline_bl_portfolio(self) -> PortfolioMetrics:
        """
        Backtest baseline Black-Litterman strategy using market equilibrium.

        Implementation Notes:
        --------------------
        - Creates neutral views by setting up a minimal view matrix (Q) and
        picking matrix (P) with very high uncertainty (omega)
        - This effectively makes the views negligible, allowing market
        equilibrium to dominate the posterior distribution
        - Uses market-implied returns (pi="market") as the prior

        Technical Details:
        -----------------
        - Q: Set to market equilibrium returns with single view
        - P: Identity picking matrix for the single view
        - omega: Very large uncertainty to minimize view impact
        - tau: Standard 0.05 setting for weight-on-views scalar
        """
        # Convert market caps to use ticker symbols for consistency
        ticker_mcaps = {
            self.publishers[pub]: mcap for pub, mcap in self.market_caps.items()
        }

        # Create minimal neutral view
        # We'll use the first asset as reference with zero view
        # The high uncertainty (omega) will make this view negligible
        Q = np.array([0.0])  # Single neutral view
        P = np.zeros((1, len(self.publishers)))  # Single row picking matrix
        P[0, 0] = 1  # Reference first asset
        omega = np.array([[100.0]])  # High uncertainty to minimize view impact

        # Initialize Black-Litterman model with neutral view
        bl = BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            pi="market",  # Use market-implied equilibrium returns
            market_caps=ticker_mcaps,
            risk_aversion=self.risk_aversion,
            Q=Q,  # Add minimal view matrix
            P=P,  # Add picking matrix
            omega=omega,  # High uncertainty
            tau=0.05,  # Standard weight-on-views scalar
        )

        # Rest of the implementation remains the same...
        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()

        ef = EfficientFrontier(ret_bl, cov_bl)
        ef.max_sharpe(risk_free_rate=self.weekly_rf_rate)
        weights = pd.Series(ef.clean_weights())

        # Calculate returns using these weights
        portfolio_returns = []
        valid_dates = []
        for date in self.returns.index[:-1]:
            next_returns = self.returns.shift(-1).loc[date]
            portfolio_return = (weights * next_returns).sum()
            portfolio_returns.append(portfolio_return)
            valid_dates.append(date)

        portfolio_returns = pd.Series(portfolio_returns, index=valid_dates)

        # Calculate metrics
        cum_returns = (1 + portfolio_returns).cumprod()
        excess_returns = portfolio_returns - self.weekly_rf_rate
        sharpe = np.sqrt(52) * excess_returns.mean() / excess_returns.std()
        drawdown = (cum_returns / cum_returns.cummax() - 1).min()

        return PortfolioMetrics(
            cumulative_returns=cum_returns, sharpe_ratio=sharpe, max_drawdown=drawdown
        )

    def construct_views(
        self,
        predictions: pd.Series,
        probabilities: pd.Series,
        barriers: pd.DataFrame,
        date: pd.Timestamp,
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Construct Black-Litterman views from model predictions

        Args:
            predictions: Series of +1/-1 predictions for each publisher
            probabilities: Series of model confidence scores
            barriers: DataFrame containing take profit/stop loss levels
            date: Current rebalancing date

        Returns:
            views_dict: Dictionary mapping publishers to expected returns
            omega: Uncertainty matrix based on model probabilities
        """
        views_dict = {}
        uncertainties = []

        for publisher in self.publishers.keys():
            print(predictions)
            print(probabilities)
            pred = predictions[publisher]
            prob = probabilities[publisher]
            pub_barriers = barriers.loc[pd.IndexSlice[:, publisher], :]
            print(pub_barriers)
            print(pub_barriers.index)
            print(pub_barriers.index.get_level_values(0))
            print(date)
            target_date = pd.Timestamp(date.date())
            pub_barriers.index = pd.MultiIndex.from_arrays(
                [
                    pd.DatetimeIndex(pub_barriers.index.get_level_values(0))
                    .tz_convert("UTC")
                    .tz_localize(None)
                    .normalize(),
                    pub_barriers.index.get_level_values("publisher"),
                ]
            )
            print(target_date)
            print(pub_barriers)
            # Convert prediction to view using appropriate barrier
            if pred == 1:
                view = pub_barriers.loc[(target_date, publisher), "take_profit"]
                print(view)
                confidence = prob
                print(f"CONFIDENCE: {confidence}")
            else:
                view = pub_barriers.loc[(target_date, publisher), "stop_loss"]
                print(view)
                confidence = 1 - prob
                print(f"CONFIDENCE: {confidence}")

            views_dict[publisher] = view
            # Higher uncertainty (lower confidence) -> larger variance
            print(f"UNCERTAINTY: {(1 - confidence) ** 2}")
            uncertainties.append((1 - confidence) ** 2)
        print(f"UNCERTAINTIES: {uncertainties}")
        omega = np.diag(uncertainties)
        return views_dict, omega

    def optimize_portfolio(
        self, views_dict: Dict[str, float], omega: np.ndarray
    ) -> pd.Series:
        """
        Optimize portfolio weights using Black-Litterman model
        """
        print(self.cov_matrix)
        print(self.market_caps)
        print(views_dict)
        print(omega)
        # Convert publisher names to tickers in views
        ticker_views = {self.publishers[pub]: view for pub, view in views_dict.items()}

        # Convert market caps to use ticker symbols
        ticker_mcaps = {
            self.publishers[pub]: mcap for pub, mcap in self.market_caps.items()
        }
        print(self.risk_aversion)
        bl = BlackLittermanModel(
            cov_matrix=self.cov_matrix,
            pi="market",
            market_caps=ticker_mcaps,
            risk_aversion=self.risk_aversion,
            absolute_views=ticker_views,
            omega=omega,
        )

        ret_bl = bl.bl_returns()
        cov_bl = bl.bl_cov()
        print(ret_bl)
        print(cov_bl)

        # Optimize for maximum Sharpe ratio
        ef = EfficientFrontier(ret_bl, cov_bl)
        ef.max_sharpe(risk_free_rate=self.weekly_rf_rate)
        weights = pd.Series(ef.clean_weights())

        return weights

    def backtest_strategy(
        self,
        predictions_df: pd.DataFrame,
        probabilities_df: pd.DataFrame,
        barriers_df: pd.DataFrame,
    ) -> PortfolioMetrics:
        """
        Backtest Black-Litterman strategy using neural network predictions

        Args:
            predictions_df: DataFrame of model predictions (+1/-1)
            probabilities_df: DataFrame of model confidence scores
            barriers_df: DataFrame of take profit/stop loss levels

        Returns:
            PortfolioMetrics containing performance statistics
        """
        portfolio_returns = []
        weights_history = []
        print("preds and probs")
        print(predictions_df)
        print(probabilities_df)

        for date in predictions_df.index:
            # Construct views for current date
            views_dict, omega = self.construct_views(
                predictions_df.loc[date], probabilities_df.loc[date], barriers_df, date
            )

            # Optimize portfolio
            weights = self.optimize_portfolio(views_dict, omega)
            weights_history.append(weights)

            # Calculate portfolio return
            next_returns = self.returns.shift(-1).loc[date]
            portfolio_return = (weights * next_returns).sum()
            portfolio_returns.append(portfolio_return)

        portfolio_returns = pd.Series(portfolio_returns, index=predictions_df.index)
        weights_history = pd.DataFrame(weights_history, index=predictions_df.index)

        # Calculate metrics
        cum_returns = (1 + portfolio_returns).cumprod()
        excess_returns = portfolio_returns - self.weekly_rf_rate
        sharpe = np.sqrt(52) * excess_returns.mean() / excess_returns.std()
        drawdown = (cum_returns / cum_returns.cummax() - 1).min()

        metrics = PortfolioMetrics(
            cumulative_returns=cum_returns, sharpe_ratio=sharpe, max_drawdown=drawdown
        )

        # Visualize final weights
        self.plot_weights(weights_history.iloc[-1])

        return metrics

    def plot_weights(self, weights: pd.Series):
        """Plot final portfolio weights as a pie chart"""
        plt.figure(figsize=(10, 10))
        plt.pie(weights, labels=weights.index, autopct="%1.1f%%")
        plt.title("Final Portfolio Weights")
        plt.axis("equal")
        plt.savefig("portfolio_weights.png")
        plt.close()


class PortfolioReportFormatter:
    """Format and save portfolio performance metrics"""

    def __init__(self):
        self.results = {}

    def add_result(self, publisher_set: str, metrics: PortfolioMetrics):
        self.results[publisher_set] = {
            "Cumulative Return": metrics.cumulative_returns.iloc[-1],
            "Sharpe Ratio": metrics.sharpe_ratio,
            "Maximum Drawdown": metrics.max_drawdown,
        }

    def save_report(self, filename: str):
        with open(filename, "w") as f:
            f.write("Black-Litterman Portfolio Performance Report\n")
            f.write("-" * 50 + "\n\n")

            for publisher_set, metrics in self.results.items():
                f.write(f"Publisher Set: {publisher_set}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
