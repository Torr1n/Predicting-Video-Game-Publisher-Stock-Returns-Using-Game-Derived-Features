import numpy as np
import pandas as pd
from feature_engineering.helpers import (
    calculate_engagement_efficiency,
    calculate_engagement_quality,
    calculate_genre_herf_index,
    calculate_lifecycle_efficiency,
    calculate_player_base_stability,
    calculate_player_growth_rate,
    calculate_purchase_intent_momentum,
    calculate_quality_adjusted_engagement,
    calculate_quality_adjusted_value,
    calculate_retention_strength,
    calculate_revenue_concentration,
    calculate_revenue_per_active_player_hour,
    calculate_revenue_predictability,
    calculate_revenue_sustainability,
    calculate_sentiment_divergence,
    calculate_social_momentum,
    calculate_weighted_sentiment_momentum,
    calculate_wishlist_conversion_rate,
    publisher_base_aggregate_features,
)


def generate_publisher_agg_features(games):
    all_dates = pd.DatetimeIndex([])
    for game in games:
        all_dates = all_dates.union(game.features.index)
    all_dates = all_dates.sort_values()
    valid_games = [game for game in games if game.features is not None]
    # For all features calculate:
    # Revenue weighted means and standard deviations
    # For basic features calcuate:
    # Age-adjusted AND revenue weighted mean
    # For genre relative features calculate:
    # Revenue from outperforming games
    features = publisher_base_aggregate_features(valid_games, all_dates)

    # Diversity Measures
    features["genre_herf_index"] = calculate_genre_herf_index(valid_games, all_dates)
    features["revenue_concentration"] = calculate_revenue_concentration(
        valid_games, all_dates
    )

    # Lifecycle Metrics
    features["lifecycle_efficiency"] = calculate_lifecycle_efficiency(
        valid_games, all_dates
    )

    return features


def create_game_agg_features(daily_data):
    """Create weekly features for game in 7 categries
    1. Game Engagement Metrics: focused on the number, activity and quality of players
    2. Sentiment Metrics: focused on changes in the game's review score
    3. Monetization Metrics: focused on game revenue and sales efficiency
    4. Lifecycle-Adjusted Metrics: focused on asssessing game performance relative to its age
    5. Stabiltiy Metrics: focused on the coefficient of variation for playerbase and revenue
    6. Leading Indicator Metrics: focused on momentum of followers and wishlists which may preceed sales
    7. Value Metrics: focused on playerbase activity relative to game price and sales
    """
    weekly_data = aggregate_daily_history(daily_data)
    features = pd.DataFrame(index=weekly_data.index)
    # Game Engagement Metrics
    features["player_growth_rate"] = calculate_player_growth_rate(weekly_data)
    features["quality_adjusted_engagement"] = calculate_quality_adjusted_engagement(
        weekly_data
    )
    # Sentiment Metrics
    features["weighted_sentiment_momentum"] = calculate_weighted_sentiment_momentum(
        weekly_data
    )
    features["sentiment_divergence"] = calculate_sentiment_divergence(weekly_data)
    # Monetization Metrics
    features["revenue_per_active_player_hour"] = (
        calculate_revenue_per_active_player_hour(weekly_data)
    )
    features["wishlist_conversion_rate"] = calculate_wishlist_conversion_rate(
        weekly_data
    )
    # Lifecycle-Adjusted Metrics
    features["retention_strength"] = calculate_retention_strength(weekly_data)
    features["revenue_sustainability"] = calculate_revenue_sustainability(weekly_data)
    # Stability Metrics
    features["player_base_stability"] = calculate_player_base_stability(weekly_data)
    features["revenue_predictability"] = calculate_revenue_predictability(weekly_data)
    # Leading Indicator Metrics
    features["social_momentum"] = calculate_social_momentum(weekly_data)
    features["purchase_intent_momentum"] = calculate_purchase_intent_momentum(
        weekly_data
    )
    # Value Metrics
    features["quality_adjusted_value"] = calculate_quality_adjusted_value(weekly_data)
    features["engagement_efficiency"] = calculate_engagement_efficiency(weekly_data)

    # Base Features
    features["revenue"] = weekly_data["revenue_total"]
    features["sales"] = weekly_data["sales_total"]
    features["players"] = weekly_data["players_mean"]
    features["avgPlaytime"] = weekly_data["avgPlaytime_total"]
    features["score"] = weekly_data["score_total"]
    features["reviews"] = weekly_data["reviews_total"]
    features["price"] = weekly_data["price_total"]
    features["days_since_release"] = weekly_data["days_since_release_total"]

    return features


def calculate_incremental_features(daily_data):
    # calculate incremental change for our net/total features
    # ie, net revenue, net sales, total reviews, total followers, total wishlists, total avgPlaytime (continuously summed)
    numeric_columns = [
        "revenue",
        "sales",
        "reviews",
        "followers",
        "wishlists",
        "avgPlaytime",
    ]

    daily_data = daily_data.copy()
    # replace None values with 0 and ensure numeric type
    for col in numeric_columns:
        # convert None to NaN
        daily_data[col] = pd.to_numeric(daily_data[col].fillna(0), errors="coerce")
        # fill any NaN with 0
        daily_data[col] = daily_data[col].fillna(0)

    # calculate incremental features
    for col in numeric_columns:
        incremental_col = f"incremental_{col}"
        try:
            # calculate diff and handle first value
            daily_data[incremental_col] = daily_data[col].diff()

            # for the first value, use the actual value instead of NaN
            # this assumes the first value represents the initial state
            if len(daily_data) > 0:
                daily_data.loc[daily_data.index[0], incremental_col] = daily_data[
                    col
                ].iloc[0]

            # handle any remaining NaN values that might occur
            daily_data[incremental_col] = daily_data[incremental_col].fillna(0)

        except Exception as e:
            print(f"Warning: Error calculating incremental values for {col}: {str(e)}")
            # if calculation fails, set incremental values to 0
            daily_data[incremental_col] = 0

    return daily_data


def aggregate_daily_history(daily_data):
    # for the given daily game data, aggregate (average) across week, starting from Monday
    daily_data = calculate_incremental_features(daily_data)

    daily_data["week"] = daily_data["timeStamp"].dt.to_period("W-SUN")

    aggregation_dict = {}
    # features for which it makes sense to take the average and std dev
    mean_std_cols = [
        "incremental_reviews",
        "players",
        "incremental_sales",
        "incremental_revenue",
        "incremental_followers",
        "incremental_wishlists",
        "incremental_avgPlaytime",
    ]
    for col in mean_std_cols:
        aggregation_dict[col + "_mean"] = (col, "mean")
        aggregation_dict[col + "_std"] = (col, "std")

    # net features for which it makese sense to take the most recent value
    last_val_cols = [
        "revenue",
        "sales",
        "reviews",
        "followers",
        "wishlists",
        "avgPlaytime",
        "score",
        "days_since_release",
        "price",
    ]
    for col in last_val_cols:
        aggregation_dict[col + "_total"] = (col, "last")

    weekly_stats = daily_data.groupby("week").agg(**aggregation_dict)
    return weekly_stats
