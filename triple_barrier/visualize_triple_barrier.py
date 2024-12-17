from datetime import date
import random
from matplotlib import pyplot as plt
import numpy as np


def visualize_results(price_data, labels):
    # create a figure with 2 subplots for overall visual + indivudal label
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # plot 1: price path with labels
    plot_price_path(price_data, labels, ax1)

    # plot 2: example week with barriers
    plot_example_week(price_data, labels, ax2)

    plt.tight_layout()
    return fig


def plot_price_path(price_data, labels, ax):
    # plot the full price path with labels
    prices = price_data["Close"]

    # plot price path
    ax.plot(prices.index, prices, color="gray", alpha=0.5, label="Price")

    # plot points where barriers were touched
    for label_type, color in [(-1, "red"), (1, "green")]:
        mask = labels["label"] == label_type
        touch_times = labels[mask]["barrier_earliest"]
        if len(touch_times) > 0:
            touch_prices = prices.loc[touch_times.to_list()]
            ax.scatter(
                touch_times,
                touch_prices,
                color=color,
                label=f'{"Positive" if label_type==1 else "Negative"} Label',
                alpha=0.6,
            )

    ax.set_title("Price Path with Labels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)


def plot_example_week(price_data, labels, ax):
    # plot an example week with barriers
    # find a week with an interesting price path
    interesting_weeks = labels[labels["barrier_earliest"] != labels["vertical_barrier"]]
    if len(interesting_weeks) > 0:
        random_week = random.randrange(len(interesting_weeks))
        example_week = interesting_weeks.iloc[random_week]
    else:
        example_week = labels.iloc[71]

    # get price path for this week
    monday = example_week["first_bar_start_date"]
    friday = example_week["vertical_barrier"]
    week_prices = price_data.loc[monday:friday, "Close"]

    # calculate log returns from Monday open
    returns = np.log(week_prices / week_prices.iloc[0])
    # plot returns path
    ax.plot(returns.index, returns, color="blue", marker="o", label="Returns")

    # add horizontal lines for barriers (relative to Monday open price)
    take_profit_level = example_week["take_profit"]
    stop_loss_level = example_week["stop_loss"]

    # plot barriers
    ax.axhline(y=take_profit_level, color="g", linestyle="--", label="Take Profit")
    ax.axhline(y=stop_loss_level, color="r", linestyle="--", label="Stop Loss")
    ax.axvline(
        x=example_week["vertical_barrier"],
        color="gray",
        linestyle="--",
        label="Friday Close",
    )

    # if a barrier was touched, mark it
    if example_week["barrier_earliest"] != example_week["vertical_barrier"]:
        touch_time = example_week["barrier_earliest"]
        touch_return = example_week["returns"]
        ax.scatter(
            touch_time,
            touch_return,
            color="orange",
            s=200,
            zorder=5,
            label="Barrier Touch",
        )

    # format the plot
    week_start = monday.strftime("%Y-%m-%d")
    ax.set_title(f"Example Week (Starting {week_start})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Returns")
    ax.legend()
    ax.grid(True)

    # rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45)
