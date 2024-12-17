import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from api import (
    audience_overlap_games_get_request,
    similar_games_get_request,
)
from feature_engineering.feature_engineering import *


class AudienceOverlap:
    def __init__(self, game):
        self.base_game = game
        self.similar_games, self.similarity_scores = self.load_overlap_games(game)
        self.averages = self._aggregate_over_genre_averages()

    def load_overlap_games(self, game):
        # loads data for similar games to the given game
        json_data = audience_overlap_games_get_request(game.id)
        audience_overlap = pd.json_normalize(
            json_data["audienceOverlap"],
            errors="ignore",
        )
        audience_overlap = audience_overlap.rename(
            columns={"steamId": "id", "link": "similarity"}
        )
        similarity_scores = (
            audience_overlap[:100].set_index("id")["similarity"].to_dict()
        )
        games = []
        for id in similarity_scores.keys():
            if id == game.id:
                continue
            similar_game = Game(id)
            if similar_game != None:
                games.append(similar_game)
        return games, similarity_scores

    def _aggregate_over_genre_averages(self):
        # aggregate the weighted average and std dev of all features across games in the genre
        # get features dictionary for all games
        games_and_features = {
            game.id: create_game_agg_features(game.history)
            for game in self.similar_games
        }

        # get all unique dates
        all_dates = pd.concat(
            [df for df in games_and_features.values()], join="inner", axis=1
        ).index.unique()
        all_dates.sort_values()

        # initialize result DataFrame
        result = pd.DataFrame(index=all_dates)

        columns = next(iter(games_and_features.values())).columns

        # process all dates
        for date in result.index:
            # collect all features across games for this date
            date_features = []
            weights = []

            for game_id, features in games_and_features.items():
                if date in features.index:
                    row_data = features.loc[date]
                    if (
                        not row_data.isnull().all()
                    ):  # Check if row has any non-null values
                        date_features.append(row_data)
                        weights.append(self.similarity_scores[game_id])

            if date_features:
                # Convert to DataFrame for vectorized operations
                date_df = pd.DataFrame(date_features)
                weights = np.array(weights)
                normalized_weights = weights / np.sum(weights)

                # Calculate weighted means for all mean/total columns at once
                weighted_means = self.calculate_weighted_means_excluding_nans(
                    date_df, columns, normalized_weights
                )
                result.loc[date, columns] = weighted_means

                # Vectorized weighted std calculation
                weighted_stds = self.calculate_weighted_std_excluding_nans(
                    date_df, columns, weighted_means, normalized_weights
                )

                # Add std columns to result
                for col, std in zip(columns, weighted_stds):
                    result.loc[date, f"{col}_std"] = std

        return result

    def calculate_weighted_means_excluding_nans(
        self, date_df, columns, normalized_weights
    ):
        # calculate weighted means for each column, excluding NaN values and renormalizing weights
        weighted_means = []

        for col in columns:
            # get mask of non-NaN values for this column
            valid_mask = ~date_df[col].isna()

            if valid_mask.any():  # if we have any valid values
                # select only valid values and their corresponding weights
                valid_values = date_df.loc[valid_mask, col]
                valid_weights = normalized_weights[valid_mask]

                # renormalize weights for just the valid values
                renormalized_weights = valid_weights / valid_weights.sum()

                # calculate weighted mean for this column
                col_weighted_mean = np.average(
                    valid_values, weights=renormalized_weights
                )
                weighted_means.append(col_weighted_mean)
            else:
                # if no valid values, append NaN
                weighted_means.append(np.nan)

        return np.array(weighted_means)

    def calculate_weighted_std_excluding_nans(
        self, date_df, columns, weighted_means, normalized_weights
    ):
        # calculate weighted standard deviation for each column, excluding NaN values and renormalizing weights
        weighted_stds = []

        for col, col_mean in zip(columns, weighted_means):
            # get mask of non-NaN values for this column
            valid_mask = ~date_df[col].isna()

            if valid_mask.any():
                # select only valid values and their corresponding weights
                valid_values = date_df.loc[valid_mask, col]
                valid_weights = normalized_weights[valid_mask]

                # renormalize weights for just the valid values
                renormalized_weights = valid_weights / valid_weights.sum()

                # calculate deviations for valid values only
                deviations = valid_values - col_mean

                # calculate weighted variance then std
                weighted_var = np.average(deviations**2, weights=renormalized_weights)
                weighted_std = np.sqrt(weighted_var)
                weighted_stds.append(weighted_std)
            else:
                # if no valid values, append NaN
                weighted_stds.append(np.nan)

        return np.array(weighted_stds)


class ParallelAudienceOverlap(AudienceOverlap):
    def __init__(self, game):
        self.base_game = game
        self.similar_games = []
        self.similarity_scores = {}
        self.max_workers = min(4, mp.cpu_count() - 1)
        self._initialize_games()
        print(f"Loaded {len(self.similar_games)} games in audienceOverlap")
        self.averages = self._aggregate_over_genre_averages()

    def _initialize_games(self):
        # load overlap games
        game_ids, similarities = self._get_overlap_game_ids_and_similarity_scores()

        # create similarity score dictionary
        self.similarity_scores = dict(zip(game_ids, similarities))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # add delay between submissions to prevent API overload
            futures = []
            for game_id in game_ids:
                futures.append(executor.submit(self._initialize_single_game, game_id))
                time.sleep(0.5)

            # collect successfully initialized games
            for future in as_completed(futures):
                try:
                    game = future.result()
                    if game is not None:
                        self.similar_games.append(game)
                except Exception as e:
                    print(f"Failed to process game: {str(e)}")

    def _get_overlap_game_ids_and_similarity_scores(
        self,
    ) -> Tuple[List[int], List[float]]:
        # get list of game IDs in the games audienceOverlap
        try:
            json_data = audience_overlap_games_get_request(self.base_game.id)
            if not json_data or "audienceOverlap" not in json_data:
                print(f"No audience overlap data found for game {self.base_game.id}")
                raise Exception()
            audience_overlap = pd.json_normalize(
                json_data["audienceOverlap"],
                errors="ignore",
            )
            audience_overlap = audience_overlap.rename(
                columns={"steamId": "id", "link": "similarity"}
            )
            return (
                audience_overlap[:100]["id"].tolist(),
                audience_overlap[:100]["similarity"].tolist(),
            )
        except Exception as e:
            print(
                f"Error getting game {self.base_game.id} audience overlap IDs: {str(e)}"
            )
            return []

    @staticmethod
    def _initialize_single_game(game_id):
        # initialization of a single game
        try:
            game = Game(game_id)
            if game is None:
                print(f"Game {game_id} initialization returned None")
                return None
            return game
        except Exception as e:
            print(f"Error initializing game {game_id}: {str(e)}")
            return None


class Genre(AudienceOverlap):
    # aggregate stats across games with similar genres, tags and release date to the given game
    def __init__(self, game):
        self.base_game = game
        self.similar_games = self.load_similar_games(game)
        # if we found less than 30 games in the genre, broaden our search
        # we want to estimate genre means and std devs to calcualte a z score for our game compared to the genre
        # so we need at least 30 games for our distribution to be approximately normal by CLT
        if len(self.similar_games) < 30:
            self.games = self.load_similar_games(game, broad=True)
        self.similarity_scores = self._calculate_similarity_scores()
        self.averages = self._aggregate_over_genre_averages()

    def load_similar_games(self, game, broad=False):
        # loads data for similar games to the given game
        if broad:
            json_data = similar_games_get_request(
                game.genres[:1], game.tags[:2], game.release_date
            )
        else:
            json_data = similar_games_get_request(
                game.genres[:2], game.tags[:3], game.release_date, broad=True
            )
        ids = pd.json_normalize(json_data["result"][:100])
        games = []
        for id in ids.id:
            if id == game.id:
                continue
            similar_game = Game(id)
            if similar_game != None:
                games.append(similar_game)
        return games

    def _calculate_similarity_scores(self):
        # calculate similarity scores between base game and all similar games
        GENRE_WEIGHT = 0.6
        TAG_WEIGHT = 0.4

        similarity_scores = {}
        base_genres = set(self.base_game.genres)
        base_tags = set(self.base_game.tags)

        for game in self.similar_games:
            # Calculate genre Jaccard similarity
            game_genres = set(game.genres)
            genre_similarity = (
                len(base_genres & game_genres) / len(base_genres | game_genres)
                if base_genres or game_genres
                else 0
            )

            # Calculate tag Jaccard similarity
            game_tags = set(game.tags)
            tag_similarity = (
                len(base_tags & game_tags) / len(base_tags | game_tags)
                if base_tags or game_tags
                else 0
            )

            # Combined weighted similarity score
            similarity_scores[game.id] = (
                GENRE_WEIGHT * genre_similarity + TAG_WEIGHT * tag_similarity
            )

        return similarity_scores


class ParallelGenre(Genre):
    def __init__(self, game):
        self.base_game = game
        self.similar_games = []
        self.max_workers = min(4, mp.cpu_count() - 1)
        self._initialize_games()
        print(f"Loaded {len(self.similar_games)} games in genre")
        self.similarity_scores = self._calculate_similarity_scores()
        self.averages = self._aggregate_over_genre_averages()

    def _initialize_games(self):
        # initialize similar games in parallel
        game_ids = self._get_similar_game_ids()
        if not game_ids:
            return
        # if we found less than 30 games, try broader search
        if len(game_ids) < 30:
            game_ids = self._get_similar_game_ids(broad=True)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for game_id in game_ids:
                if game_id != self.base_game.id:  # skip the base game
                    futures.append(
                        executor.submit(self._initialize_single_game, game_id)
                    )
                    time.sleep(0.5)

            # collect results
            for future in as_completed(futures):
                try:
                    game = future.result()
                    if game is not None:
                        self.similar_games.append(game)
                except Exception as e:
                    print(f"Failed to process genre similar game: {str(e)}")

    def _get_similar_game_ids(self, broad=False) -> List[int]:
        # get list of similar game IDs
        try:
            if broad:
                json_data = similar_games_get_request(
                    self.base_game.genres[:1],
                    self.base_game.tags[:2],
                    self.base_game.release_date,
                    broad=True,
                )
            else:
                json_data = similar_games_get_request(
                    self.base_game.genres[:2],
                    self.base_game.tags[:3],
                    self.base_game.release_date,
                )

            if not json_data or "result" not in json_data:
                print(f"No similar games found for game {self.base_game.id}")
                return []

            return pd.json_normalize(json_data["result"])["id"].tolist()

        except Exception as e:
            print(f"Error getting similar games for game {self.base_game.id}: {str(e)}")
            return []

    @staticmethod
    def _initialize_single_game(game_id: int):
        # initialization of a single game
        try:
            game = Game(game_id)
            if game is None:
                print(f"Game {game_id} initialization returned None")
                return None
            return game
        except Exception as e:
            print(f"Error initializing game {game_id}: {str(e)}")
            return None


from publisher import Game
