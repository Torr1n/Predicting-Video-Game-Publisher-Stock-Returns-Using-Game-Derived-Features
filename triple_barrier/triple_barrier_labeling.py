import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pytz
from datetime import datetime, time, timedelta
import yfinance as yf
from dateutil.relativedelta import relativedelta


class ExchangeInfo:
    # Exchange trading hours information using yfinance ticker.info attributes

    # standard trading hours by exchange
    TRADING_HOURS = {
        "JPX": {"open": time(9, 0), "close": time(15, 0)},  # Japanese Exchange
        "NYQ": {"open": time(9, 30), "close": time(16, 0)},  # NYSE
        "NMS": {"open": time(9, 30), "close": time(16, 0)},  # NASDAQ
        "PAR": {"open": time(9, 0), "close": time(17, 30)},  # Euronext Paris
        "LSE": {"open": time(8, 0), "close": time(16, 30)},  # London Stock Exchange
        "WSE": {"open": time(9, 0), "close": time(16, 50)},  # Warsaw Stock Exchange
    }

    @classmethod
    def get_exchange_info(cls, ticker_symbol):
        # using yfinance ticker info attributes, return exchange information including timezone and trading hours

        try:
            # get exchange information directly from yfinance
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            exchange_info = {
                "timezone": info.get("timeZoneFullName"),
                "timezone_short": info.get("timeZoneShortName"),
                "gmt_offset": info.get("gmtOffSetMilliseconds", 0)
                / 3600000,  # convert to hours
                "exchange": info.get("exchange"),
            }

            # get trading hours for the exchange
            trading_hours = cls.TRADING_HOURS.get(
                exchange_info["exchange"],
                cls.TRADING_HOURS["NYQ"],  # default to NYSE hours if unknown
            )

            exchange_info["trading_hours"] = trading_hours

            return exchange_info

        except Exception as e:
            print(f"Error getting exchange info for {ticker_symbol}: {str(e)}")
            # return NYSE as default with UTC timezone
            return {
                "timezone": "America/New_York",
                "timezone_short": "EST",
                "gmt_offset": -5,
                "exchange": "NYQ",
                "trading_hours": cls.TRADING_HOURS["NYQ"],
            }


def fetch_hourly_data_in_chunks(ticker, start_date, end_date, chunk_size=60):
    # fetch hourly data in chunks to overcome yfinance limitations on hourly data
    all_data = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=chunk_size), end_date)

        # print(f"Fetching chunk from {current_start.date()} to {current_end.date()}")
        chunk = yf.download(
            ticker, start=current_start, end=current_end, interval="1h", progress=False
        )

        if not chunk.empty:
            all_data.append(chunk)

        current_start = current_end

    if not all_data:
        raise ValueError("No data was fetched")

    return pd.concat(all_data)


def get_ticker_data(ticker):
    # get stock data with yfinance
    # define date range
    end_date = datetime.now()
    start_date = end_date - relativedelta(days=729)

    # get hourly data in chunks
    try:
        df_hourly = fetch_hourly_data_in_chunks(ticker, start_date, end_date)
    except Exception as e:
        print(f"Error fetching hourly data: {str(e)}")
        return None

    # remove timezone information
    df_hourly.index = df_hourly.index.tz_localize(None)

    return df_hourly


def get_weekly_vertical_barriers(price_data, exchange_info, start_date, end_date):
    # generate weekly vertical time barriers at Friday close, handling different exchanges

    # create weekly events starting from Mondays
    monday_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="W-MON",
    )
    # adjust Monday times to account for exchange opening times
    trading_hours = exchange_info["trading_hours"]
    monday_open_dates = [
        monday.replace(
            hour=trading_hours["open"].hour, minute=trading_hours["open"].minute
        )
        for monday in monday_dates
    ]
    # initialize barriers
    barriers_df = pd.DataFrame(index=monday_open_dates)
    for monday in monday_open_dates:
        # set to market close time using exchange-specific hours
        friday_open = monday + pd.Timedelta(days=4)
        friday_close = friday_open.replace(
            hour=trading_hours["close"].hour, minute=trading_hours["close"].minute
        )
        # find the actual last trading bar of the week
        week_data = price_data.loc[monday:friday_close]
        if len(week_data) > 0:
            last_bar = week_data.index[-1]
            barriers_df.loc[monday, "vertical_barrier"] = last_bar

    return barriers_df.dropna()


def calculate_weekly_horizontal_barriers(
    price_data, exchange_info, start_date, end_date, volatility_window=5
):
    # calculate dynamic take profit and stop loss barriers based on weekly volatility
    # resample to weekly to handle irregular trading hours
    weekly_log_prices = np.log(price_data["Close"].resample("W-MON").last())
    weekly_returns = weekly_log_prices.diff()
    # calculate rolling volatility
    volatility = weekly_returns.rolling(window=volatility_window).std()

    # set dynamic barriers
    take_profit_multiplier = 1.5 * volatility
    stop_loss_multiplier = -1.5 * volatility

    weekly_log_prices["take_profit"] = take_profit_multiplier.iloc[:, 0].bfill()

    weekly_log_prices["stop_loss"] = stop_loss_multiplier.iloc[:, 0].bfill()

    # adjust for timezone alignment with start and end dates
    trading_hours = exchange_info["trading_hours"]
    weekly_log_prices.index = weekly_log_prices.index.map(
        lambda date: date.replace(
            hour=trading_hours["open"].hour, minute=trading_hours["open"].minute
        )
    )
    # filter results to be within desired dates
    # (price data starts before at our desired start because we need a rolling window)
    return (
        weekly_log_prices[
            (weekly_log_prices.index >= start_date)
            & (weekly_log_prices.index <= end_date)
        ]["take_profit"],
        weekly_log_prices[
            (weekly_log_prices.index >= start_date)
            & (weekly_log_prices.index <= end_date)
        ]["stop_loss"],
    )


def get_barrier_hits(price_data, barriers_df):
    # find the first barrier hit for each week based on the path of hourly price movements
    upper_barrier = barriers_df["take_profit"]
    lower_barrier = barriers_df["stop_loss"]
    barriers = barriers_df[["vertical_barrier", "take_profit", "stop_loss"]].copy()
    for week_start_date, week_end_date in (
        barriers_df["vertical_barrier"].fillna(price_data["Close"].index[-1]).items()
    ):
        first_bar = price_data["Close"].loc[week_start_date:week_end_date].iloc[0]
        first_price = first_bar.values[0]
        barriers.loc[week_start_date, "first_bar_start_date"] = first_bar.name

        # find the path of prices for this event excluding first price (price at time of event)
        path_prices = price_data["Close"].loc[week_start_date:week_end_date].iloc[1:]

        lower_breach = None
        upper_breach = None

        for timestamp, row in path_prices.iterrows():
            # Check lower barrier breach
            price = row.iloc[0]
            returns = np.log(price / first_price)
            if returns <= lower_barrier[week_start_date]:
                lower_breach = timestamp
                break

        for timestamp, row in path_prices.iterrows():
            # Check upper barrier breach
            price = row.iloc[0]
            returns = np.log(price / first_price)
            if returns >= upper_barrier[week_start_date]:
                upper_breach = timestamp
                break

        barriers.loc[week_start_date, "lower_barrier_earliest"] = lower_breach
        barriers.loc[week_start_date, "upper_barrier_earliest"] = upper_breach
        barrier_times = [
            time
            for time in [lower_breach, upper_breach, week_end_date]
            if pd.notna(time)
        ]
        earliest_barrier = min(barrier_times)
        barriers.loc[week_start_date, "barrier_earliest"] = earliest_barrier

        # calculate weight based off distance to closest label
        if earliest_barrier == week_end_date:
            # vertical barrier hit first
            final_price = (
                price_data["Close"]
                .loc[week_start_date:week_end_date]
                .iloc[-1]
                .values[0]
            )

            returns = np.log(final_price / first_price)

            take_profit_threshold = upper_barrier[week_start_date]
            stop_loss_threshold = lower_barrier[week_start_date]

            if returns >= 0:
                # take profit
                weight = min(1, returns / take_profit_threshold)
            else:
                # stop loss
                weight = min(1, abs(returns / stop_loss_threshold))
        else:
            weight = 1
        barriers.loc[week_start_date, "weight"] = weight
    return barriers


def get_log_returns(price_data, earliest_touch_df):
    # calculates the log returns between the event's start time and first barrier hit
    earliest_touch_df["returns"] = np.log(
        price_data["Close"].loc[earliest_touch_df["barrier_earliest"].to_list()].values
        / price_data["Close"]
        .loc[earliest_touch_df["first_bar_start_date"].to_list()]
        .values
    )
    returns = earliest_touch_df["returns"]
    return earliest_touch_df


def get_classification_labels(price_data, earliest_touch_df):
    # binary classifaction as described by Lopez de Prado (-1 or 1)
    returns_df = get_log_returns(price_data, earliest_touch_df)
    returns_df["label"] = np.sign(returns_df["returns"])
    return returns_df


def create_weekly_triple_barrier_labels(
    ticker_symbol, start_date="2023-01-15", end_date="2025-01-07"
):
    # apply triple barrier labeling to weekly periods for any exchange

    # get hourly price data
    price_data = get_ticker_data(ticker_symbol)
    # get exchange information using built-in yfinance attributes
    exchange_info = ExchangeInfo.get_exchange_info(ticker_symbol)
    exchange_tz = pytz.timezone(exchange_info["timezone"])

    # convert to local timezone for exchange
    start_date = (
        pd.Timestamp(datetime.strptime(start_date, "%Y-%m-%d"))
        .tz_localize("UTC")
        .tz_convert(exchange_tz)
    )
    end_date = (
        pd.Timestamp(datetime.strptime(end_date, "%Y-%m-%d"))
        .tz_localize("UTC")
        .tz_convert(exchange_tz)
    )

    price_data_localized = price_data.tz_localize("UTC").tz_convert(exchange_tz)
    print(price_data_localized)
    # get weekly barrier times
    barriers_df = get_weekly_vertical_barriers(
        price_data_localized, exchange_info, start_date, end_date
    )
    # calculate dynamic barriers
    take_profit, stop_loss = calculate_weekly_horizontal_barriers(
        price_data_localized, exchange_info, start_date, end_date
    )
    # add horizontal barriers levels to our barriers df
    barriers_df["take_profit"] = take_profit
    barriers_df["stop_loss"] = stop_loss

    # generate labels
    earliest_touch_df = get_barrier_hits(price_data_localized, barriers_df)
    labels = get_classification_labels(price_data_localized, earliest_touch_df)
    print(labels["take_profit"].values)
    print(labels["stop_loss"].values)
    return labels, price_data_localized
