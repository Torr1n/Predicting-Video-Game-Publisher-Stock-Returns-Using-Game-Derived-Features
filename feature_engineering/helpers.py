import numpy as np
import pandas as pd


def safe_divide(a, b):
    # safely divide two series returning 0 when denominator is 0
    return np.where(b != 0, a / b, 0)


# don't expect to really need this function but just in case of weird 0 values for reviews, followers or wishlists
def handle_inf_nan(series):
    # replace inf and -inf with nan, then forward fill nans
    return (
        series.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    )  # fill any remaining nans at the start with 0


def calculate_player_growth_rate(weekly_data):
    """Weekly percentage change in average daily players

    for changes from zero players, instead of forward filling, we use
    the new value normalized by the historical mean of non-zero values.
    This captures the magnitude of recovery from zero while keeping the scale
    similar to percentage changes.
    """
    # calculate regular percent changes first
    pct_change = weekly_data["players_mean"].pct_change()

    # find periods where previous value was 0 (causing infinity)
    prev_zero_mask = weekly_data["players_mean"].shift(1) == 0

    # for these periods, calculate mean of all non-zero values up to that point
    # and express the new value as a ratio to this mean
    running_nonzero_mean = (
        weekly_data["players_mean"].replace(0, np.nan).expanding().mean()
    )
    recovery_ratio = safe_divide(weekly_data["players_mean"], running_nonzero_mean)

    # use recovery ratio where we had zero previous values, else use regular pct change
    return np.where(prev_zero_mask, recovery_ratio - 1, pct_change)


def calculate_quality_adjusted_engagement(weekly_data):
    # average daily players weighted by average daily playtime and review score
    return (
        weekly_data["players_mean"]
        * abs(weekly_data["incremental_avgPlaytime_mean"])
        * (weekly_data["score_total"] / 100)
    )


def calculate_weighted_sentiment_momentum(weekly_data):
    # change in review score weighted by log of review volume
    weighted_sentiment = weekly_data["score_total"] * np.log1p(
        weekly_data["reviews_total"]
    )
    return handle_inf_nan(weighted_sentiment.pct_change(fill_method=None))


def calculate_sentiment_divergence(weekly_data, short_window=4, long_window=8):
    # difference between short-term and long-term player sentiment trends
    return (
        weekly_data["score_total"].rolling(short_window).mean()
        - weekly_data["score_total"].rolling(long_window).mean()
    )


def calculate_revenue_per_active_player_hour(weekly_data):
    # weekly revenue growth divided by player activity
    revenue_change = weekly_data["revenue_total"].diff()
    player_activity_change = (
        weekly_data["players_mean"] * weekly_data["avgPlaytime_total"]
    )
    return safe_divide(revenue_change, player_activity_change)


def calculate_wishlist_conversion_rate(weekly_data):
    # new sales divided by previous week's wishlist count
    new_sales = weekly_data["sales_total"].diff()
    prev_wishlists = weekly_data["wishlists_total"].shift(1)
    return safe_divide(new_sales, prev_wishlists)


def calculate_retention_strength(weekly_data):
    # actual players divided by expected players based on sales and game age
    expected_players = weekly_data["sales_total"] / (
        1 + np.sqrt(abs(weekly_data["days_since_release_total"]))
    )
    acutal_players = weekly_data["players_mean"]
    return safe_divide(acutal_players, expected_players)


def calculate_revenue_sustainability(weekly_data):
    # actual average daily revenue change divided by expected average daily revenue
    expected_daily_revenue = weekly_data["revenue_total"] / (
        weekly_data["days_since_release_total"] ** 2
    )
    return safe_divide(weekly_data["incremental_revenue_mean"], expected_daily_revenue)


def calculate_player_base_stability(weekly_data):
    # coefficient of variation of daily player count
    return safe_divide(weekly_data["players_std"], weekly_data["players_mean"])


def calculate_revenue_predictability(weekly_data):
    # 4-week rolling coefficient of variation of daily revenue changes
    mean_4w = weekly_data["incremental_revenue_mean"].rolling(4).mean()
    return safe_divide(weekly_data["incremental_revenue_std"], mean_4w)


def calculate_social_momentum(weekly_data, short_window=4, long_window=8):
    # normalized difference between short-term and long-term follower trends
    short_term_growth = handle_inf_nan(
        weekly_data["followers_total"]
        .pct_change(short_window)
        .rolling(window=short_window)
        .mean()
    )
    long_term_growth = handle_inf_nan(
        weekly_data["followers_total"]
        .pct_change(long_window)
        .rolling(window=long_window)
        .mean()
    )
    momentum = short_term_growth - long_term_growth
    return safe_divide(momentum, np.log1p(weekly_data["followers_total"]))


def calculate_purchase_intent_momentum(weekly_data, short_window=4, long_window=8):
    # normalized difference between short-term and long-term wishlist trends
    short_term_growth = handle_inf_nan(
        weekly_data["wishlists_total"]
        .pct_change(short_window)
        .rolling(window=short_window)
        .mean()
    )
    long_term_growth = handle_inf_nan(
        weekly_data["wishlists_total"]
        .pct_change(long_window)
        .rolling(window=long_window)
        .mean()
    )
    momentum = short_term_growth - long_term_growth
    return safe_divide(momentum, np.log1p(weekly_data["wishlists_total"]))


def calculate_quality_adjusted_value(weekly_data):
    # review score * average total playtime / price
    value = weekly_data["score_total"] * weekly_data["avgPlaytime_total"]
    return safe_divide(value, weekly_data["price_total"])


def calculate_engagement_efficiency(weekly_data):
    # geometric mean of daily playtime change divided by price and normalized average daily sales growth
    engagement = safe_divide(
        weekly_data["incremental_avgPlaytime_mean"], weekly_data["price_total"]
    )
    recent_sales_proportion = safe_divide(
        weekly_data["incremental_sales_mean"], weekly_data["sales_total"]
    )
    return np.sqrt(abs(engagement * recent_sales_proportion))


def calculate_genre_herf_index(games, dates):
    # calculate genre-level Herfindahl index over time.
    genre_herf = pd.Series(index=dates, dtype=float)

    for date in dates:
        # get revenue for each game on this date
        genre_revenues = {}
        total_revenue = 0

        # aggregate revenues by genre
        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                if pd.notnull(revenue):
                    total_revenue += revenue
                    # a game can belong to multiple genres
                    for genre in game.genres:
                        genre_revenues[genre] = genre_revenues.get(genre, 0) + revenue

        # calculate Herfindahl index if we have revenue data
        if total_revenue > 0:
            # calculate squared market shares
            squared_shares = [
                (rev / total_revenue) ** 2 for rev in genre_revenues.values()
            ]
            genre_herf[date] = sum(squared_shares)
        else:
            genre_herf[date] = np.nan

    return genre_herf


def calculate_revenue_concentration(games, dates):
    # calculate game-level revenue concentration over time.
    revenue_concentration = pd.Series(index=dates, dtype=float)

    for date in dates:
        # get revenue for each game on this date
        game_revenues = []
        total_revenue = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                if pd.notnull(revenue):
                    game_revenues.append(revenue)
                    total_revenue += revenue

        # calculate Herfindahl index if we have revenue data
        if total_revenue > 0:
            # calculate squared market shares
            squared_shares = [(rev / total_revenue) ** 2 for rev in game_revenues]
            revenue_concentration[date] = sum(squared_shares)
        else:
            revenue_concentration[date] = np.nan

    return revenue_concentration


def calculate_engagement_quality(games, dates):
    # calculate portfolio-wide engagement quality score.
    engagement_quality = pd.Series(index=dates, dtype=float)

    for date in dates:
        total_revenue = 0
        weighted_engagement = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                engagement = game.features.loc[date, "quality_adjusted_engagement"]
                price = game.features.loc[date, "price"]

                if (
                    pd.notnull(revenue)
                    and pd.notnull(engagement)
                    and pd.notnull(price)
                    and price > 0
                ):
                    total_revenue += revenue
                    # Normalize engagement by price point
                    weighted_engagement += revenue * (engagement / price)

        if total_revenue > 0:
            engagement_quality[date] = weighted_engagement / total_revenue
        else:
            engagement_quality[date] = np.nan

    return engagement_quality


def calculate_lifecycle_efficiency(games, dates):
    # calculate portfolio-wide lifecycle management efficiency.
    lifecycle_efficiency = pd.Series(index=dates, dtype=float)

    for date in dates:
        total_revenue = 0
        weighted_efficiency = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                retention = game.features.loc[date, "retention_strength"]
                sustainability = game.features.loc[date, "revenue_sustainability"]

                if (
                    pd.notnull(revenue)
                    and pd.notnull(retention)
                    and pd.notnull(sustainability)
                ):
                    total_revenue += revenue
                    weighted_efficiency += revenue * (retention * sustainability)

        if total_revenue > 0:
            lifecycle_efficiency[date] = weighted_efficiency / total_revenue
        else:
            lifecycle_efficiency[date] = np.nan

    return lifecycle_efficiency


def publisher_base_aggregate_features(games, dates):
    # generate comprehensive aggregate features across publisher portfolio.
    features = pd.DataFrame(index=dates)

    # Basic Features
    base_features = [
        "player_growth_rate",
        "quality_adjusted_engagement",
        "weighted_sentiment_momentum",
        "sentiment_divergence",
        "revenue_per_active_player_hour",
        "wishlist_conversion_rate",
        "retention_strength",
        "revenue_sustainability",
        "player_base_stability",
        "revenue_predictability",
        "social_momentum",
        "purchase_intent_momentum",
        "quality_adjusted_value",
        "engagement_efficiency",
        "revenue",
        "sales",
        "players",
        "avgPlaytime",
        "score",
        "reviews",
        "price",
    ]
    # Relative Performance Features
    relative_features = [
        "audience_overlap_relative_player_growth_rate",
        "audience_overlap_relative_quality_adjusted_engagement",
        "audience_overlap_relative_weighted_sentiment_momentum",
        "audience_overlap_relative_sentiment_divergence",
        "audience_overlap_relative_revenue_per_active_player_hour",
        "audience_overlap_relative_wishlist_conversion_rate",
        "audience_overlap_relative_retention_strength",
        "audience_overlap_relative_revenue_sustainability",
        "audience_overlap_relative_player_base_stability",
        "audience_overlap_relative_revenue_predictability",
        "audience_overlap_relative_social_momentum",
        "audience_overlap_relative_purchase_intent_momentum",
        "audience_overlap_relative_quality_adjusted_value",
        "audience_overlap_relative_engagement_efficiency",
        "audience_overlap_relative_revenue",
        "audience_overlap_relative_sales",
        "audience_overlap_relative_players",
        "audience_overlap_relative_avgPlaytime",
        "audience_overlap_relative_score",
        "audience_overlap_relative_reviews",
        "audience_overlap_relative_price",
        "genre_relative_player_growth_rate",
        "genre_relative_quality_adjusted_engagement",
        "genre_relative_weighted_sentiment_momentum",
        "genre_relative_sentiment_divergence",
        "genre_relative_revenue_per_active_player_hour",
        "genre_relative_wishlist_conversion_rate",
        "genre_relative_retention_strength",
        "genre_relative_revenue_sustainability",
        "genre_relative_player_base_stability",
        "genre_relative_revenue_predictability",
        "genre_relative_social_momentum",
        "genre_relative_purchase_intent_momentum",
        "genre_relative_quality_adjusted_value",
        "genre_relative_engagement_efficiency",
        "genre_relative_revenue",
        "genre_relative_sales",
        "genre_relative_players",
        "genre_relative_avgPlaytime",
        "genre_relative_score",
        "genre_relative_reviews",
        "genre_relative_price",
    ]
    # For all features, calculate direct revenue weighted means and std deviations
    for feature in base_features + relative_features:
        # revenue-weighted mean
        features[f"portfolio_{feature}_mean"] = calculate_revenue_weighted_mean(
            games, dates, feature
        )
        # revenue-weighted std
        features[f"portfolio_{feature}_deviation"] = calculate_revenue_weighted_std(
            games, dates, feature
        )

    # age-adjusted revenue weighted means for base features
    for feature in base_features:
        features[f"portfolio_{feature}_age_adjusted"] = calculate_age_weighted_mean(
            games, dates, feature
        )

    # revenue from outperforming games for each genre-relative metric
    for feature in relative_features:
        features[f"portfolio_{feature}_strength"] = calculate_outperformance_ratio(
            games, dates, feature
        )

    return features


def calculate_revenue_weighted_mean(games, dates, feature):
    # calculate revenue-weighted mean of a feature across portfolio
    weighted_mean = pd.Series(index=dates, dtype=float)

    for date in dates:
        total_revenue = 0
        weighted_sum = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                feature_value = game.features.loc[date, feature]

                if pd.notnull(revenue) and pd.notnull(feature_value):
                    total_revenue += revenue
                    weighted_sum += revenue * feature_value

        if total_revenue > 0:
            weighted_mean[date] = weighted_sum / total_revenue
        else:
            weighted_mean[date] = np.nan

    return weighted_mean


def calculate_revenue_weighted_std(games, dates, feature):
    # calculate revenue-weighted standard deviation of a feature across portfolio
    weighted_std = pd.Series(index=dates, dtype=float)

    for date in dates:
        revenues = []
        values = []

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                value = game.features.loc[date, feature]

                if pd.notnull(revenue) and pd.notnull(value):
                    revenues.append(revenue)
                    values.append(value)

        if revenues:
            # TODO LOOK INTO
            weights = np.array(revenues) / sum(revenues)
            mean = np.average(values, weights=weights)
            variance = np.average((values - mean) ** 2, weights=weights)
            weighted_std[date] = np.sqrt(abs(variance))
        else:
            weighted_std[date] = np.nan

    return weighted_std


def calculate_outperformance_ratio(games, dates, relative_feature):
    # calculate proportion of revenue from games outperforming peers in feature categories
    outperformance = pd.Series(index=dates, dtype=float)

    for date in dates:
        total_revenue = 0
        outperform_revenue = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                relative_value = game.features.loc[date, relative_feature]

                if pd.notnull(revenue) and pd.notnull(relative_value):
                    total_revenue += revenue
                    if relative_value > 0:  # outperforming (z-score > 0)
                        outperform_revenue += revenue

        if total_revenue > 0:
            outperformance[date] = outperform_revenue / total_revenue
        else:
            outperformance[date] = np.nan

    return outperformance


def calculate_age_weighted_mean(games, dates, feature):
    # calculate age-adjusted AND revenue weighted mean of a feature
    age_weighted_mean = pd.Series(index=dates, dtype=float)

    for date in dates:
        total_weight = 0
        weighted_sum = 0

        for game in games:
            if date in game.features.index:
                revenue = game.features.loc[date, "revenue"]
                value = game.features.loc[date, feature]
                age = game.features.loc[date, "days_since_release"]

                if pd.notnull(revenue) and pd.notnull(value) and pd.notnull(age):
                    # age weight decays with sqrt of age to capture publisher reliance on old releases
                    # normalize age to years beacuse old games still can make meaningful revenue contributions
                    age_weight = 1 / np.sqrt(abs(1 + age / 365))
                    weight = revenue * age_weight
                    total_weight += weight
                    weighted_sum += weight * value

        if total_weight > 0:
            age_weighted_mean[date] = weighted_sum / total_weight
        else:
            age_weighted_mean[date] = np.nan

    return age_weighted_mean
