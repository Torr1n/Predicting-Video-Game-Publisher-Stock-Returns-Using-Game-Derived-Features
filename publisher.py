from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from pathlib import Path
import os
from datetime import datetime
import pickle
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from api import (
    game_get_request,
    publisher_games_get_request,
)
from feature_engineering.feature_engineering import *
from triple_barrier.triple_barrier_labeling import create_weekly_triple_barrier_labels
from triple_barrier.visualize_triple_barrier import visualize_results


class ParallelPublisher:
    def __init__(self, id, ticker, max_workers=None):
        self.id = id
        self.ticker = ticker
        # use CPU count - 1 to leave one core free
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.games = []
        self.features = None
        triple_barrier_labels, price_data = create_weekly_triple_barrier_labels(ticker)
        self.OHLC_data = price_data[["Open", "High", "Low", "Close"]]
        # fig = visualize_results(price_data, triple_barrier_labels)
        # plt.show()
        self.labels = triple_barrier_labels["label"]
        self.label_weights = triple_barrier_labels["weight"]
        self.take_profit_barriers = triple_barrier_labels["take_profit"]
        self.stop_loss_barriers = triple_barrier_labels["stop_loss"]

    def initialize_games(self):
        # load published games using thread pool for API calls
        game_ids = self._get_publisher_game_ids()

        # use ThreadPoolExecutor for API calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # add delay between submissions to prevent API overload
            futures = []
            for game_id in game_ids:
                futures.append(executor.submit(self._initialize_single_game, game_id))
                time.sleep(0.5)  # add delay between API calls

            # collect successfully initialized games
            for future in as_completed(futures):
                try:
                    game = future.result()
                    if game is not None:
                        self.games.append(game)
                except Exception as e:
                    print(f"Failed to process game: {str(e)}")

    def calculate_features(self):
        # calculate features for all games using process pool for CPU-intensive work
        if not self.games:
            print("No games loaded to calculate features")
            return

        # use ProcessPoolExecutor for CPU-intensive calculations
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for game in self.games:
                futures.append(executor.submit(self._calculate_game_features, game))

            # process completed calculations
            successful_games = []
            for future, game in zip(futures, self.games):
                try:
                    features = future.result()
                    if features is not None:
                        game.features = features
                        successful_games.append(game)
                except Exception as e:
                    print(f"Failed to calculate features for game {game.id}: {str(e)}")

            self.games = successful_games

        if self.games:
            # After all games are processed, generate publisher aggregates
            self.features = generate_publisher_agg_features(self.games)
        else:
            print("No games successfully processed")

    def _get_publisher_game_ids(self) -> List[int]:
        # get list of game IDs for the publisher
        try:
            json_data = publisher_games_get_request(self.id)
            if not json_data or "games" not in json_data:
                print(f"No games data found for publisher {self.id}")
                return []
            games_df = pd.json_normalize(json_data["games"], errors="ignore")
            return games_df["id"].tolist()
        except Exception as e:
            print(f"Error getting publisher game IDs: {str(e)}")
            return []

    @staticmethod
    def _initialize_single_game(game_id):
        # initialize a single game - static method for parallel processing
        try:
            game = Game(game_id)
            if game is None:
                print(f"Game {game_id} initialization returned None")
                return None
            return game
        except Exception as e:
            print(f"Error initializing game {game_id}: {str(e)}")
            return None

    @staticmethod
    def _calculate_game_features(game):
        # calculate features for a single game - static method for parallel processing
        try:
            game._calcualte_genre_relative_features()
        except Exception as e:
            print(f"Error calculating features for game {game.id}: {str(e)}")
            raise e
        try:
            game._calcualte_audience_overlap_relative_features()
            return game.features
        except Exception as e:
            print(f"Error calculating features for game {game.id}: {str(e)}")
            raise e


class PublisherPersistence:
    # Handles persistence operations for Publisher objects

    def __init__(self, base_dir="data"):
        # initialize persistence manager with a base directory for storage
        self.base_dir = Path(base_dir)
        self._ensure_directories()

    def _ensure_directories(self):
        # create necessary directory structure if it doesn't exist
        os.makedirs(self.base_dir / "publishers", exist_ok=True)

    def _get_publisher_path(self, publisher_id, ticker):
        # generate path for publisher pickle file
        filename = f"{publisher_id}_{ticker}_{datetime.now().strftime('%Y%m%d')}.pkl"
        return self.base_dir / "publishers" / filename

    def _get_latest_file(self, directory, publisher_id, ticker):
        # get the most recent file for a publisher
        pattern = f"{publisher_id}_{ticker}_*.pkl"
        files = list(directory.glob(pattern))
        return max(files, default=None, key=lambda x: x.stat().st_mtime)

    def save_publisher(self, publisher):
        # pickle and then save publisher data
        pickle_path = self._get_publisher_path(publisher.id, publisher.ticker)
        with open(pickle_path, "wb") as f:
            pickle.dump(publisher, f)
        print(f"Publisher {publisher.id} saved!")

    def load_publisher(self, publisher_id, ticker):
        # load publisher data from pickle file
        latest_pickle = self._get_latest_file(
            self.base_dir / "publishers", publisher_id, ticker
        )
        if latest_pickle:
            with open(latest_pickle, "rb") as f:
                print(f"Publisher {publisher_id} loaded!")
                return pickle.load(f)

        return None


class PersistentParallelPublisher(ParallelPublisher):
    # extension of ParallelPublisher with persistence added

    def __init__(self, id, ticker, max_workers=None):
        # initialize publisher with persistence support

        self.persistence = PublisherPersistence()

        # try to load existing publisher
        existing_publisher = self.persistence.load_publisher(id, ticker)

        if existing_publisher:
            print("Publisher already exists, loading from pickle file!")
            # copy attributes from existing publisher
            self.__dict__.update(existing_publisher.__dict__)
        else:
            # initialize as new publisher
            print(
                "Publisher does not already exist, manually calculating labels and features!"
            )
            super().__init__(id, ticker, max_workers)
            super().initialize_games()
            print(f"Loaded {len(self.games)} games")

            super().calculate_features()
            self.save()

    def calculate_features(self):
        # override calculate_features if I ever want to recalc features
        super().initialize_games()
        super().calculate_features()
        if self.features is not None:
            self.persistence.save_publisher(self)

    def save(self):
        # explicit save method
        self.persistence.save_publisher(self)


class Publisher:
    # a steam game publisher that is publicly listed on the stock exchange
    def __init__(self, id, ticker=None):
        self.id = id
        self.ticker = ticker
        triple_barrier_labels, _ = create_weekly_triple_barrier_labels(ticker)
        self.labels = triple_barrier_labels["label"]
        self.label_weights = triple_barrier_labels["weight"]

        self.games = self.load_published_games(self.id)
        for game in self.games:
            game._calcualte_genre_relative_features()
            game._calcualte_audience_overlap_relative_features()
        self.features = generate_publisher_agg_features(self.games)

    def load_published_games(self, publisher_id):
        # loads game data for the given publisher
        json_data = publisher_games_get_request(publisher_id)
        games = pd.json_normalize(
            json_data["games"],
            errors="ignore",
        )

        publisher_games = []
        for id in games["id"]:
            publisher_game = Game(id)
            if publisher_game != None:
                publisher_games.append(publisher_game)
        return publisher_games


class Game:
    # a steam game
    def __init__(self, id):
        self.id = id
        self.genre = None
        self.audience_overlap = None
        try:
            game_data = self.load_game_data(id)
            (
                self.name,
                self.genres,
                self.tags,
                self.release_date,
                self.history,
            ) = game_data
            self.features = create_game_agg_features(self.history)
        except Exception as e:
            print(f"Could not load game: {id}")
            raise
        self.features = create_game_agg_features(self.history)

    @classmethod
    def create(cls, id):
        # factory method to create game instances with proper error handling
        try:
            return cls(id)
        except Exception:
            return None

    def load_game_data(self, game_id):
        # loads game data for the given game_id
        json_data = game_get_request(game_id)
        historical_data = pd.json_normalize(
            json_data["history"],
            errors="ignore",
        )
        one_year = 31556926000
        if (
            historical_data["timeStamp"][0]
            > json_data["firstReleaseDate"] + 5 * one_year
        ):
            raise Exception("No game data within 5 year of firstReleaseDate")
        historical_data["timeStamp"] = pd.to_datetime(
            historical_data["timeStamp"], unit="ms"
        )
        # add days since release
        release_date = pd.to_datetime(json_data["firstReleaseDate"], unit="ms")
        historical_data["days_since_release"] = (
            historical_data["timeStamp"] - release_date
        ).dt.days
        # check for and handle missing columns
        if "revenue" not in historical_data.columns:
            historical_data["revenue"] = 0
        if "followers" not in historical_data.columns:
            historical_data["followers"] = 0
        if "wishlists" not in historical_data.columns:
            historical_data["wishlists"] = 0
        if "score" not in historical_data.columns:
            historical_data["score"] = 50
        if "price" not in historical_data.columns:
            historical_data["price"] = 0
        if "avgPlaytime" not in historical_data.columns:
            historical_data["avgPlaytime"] = 0
        return (
            json_data["name"],
            json_data["genres"],
            json_data["tags"],
            json_data["firstReleaseDate"],
            historical_data,
        )

    def _calcualte_genre_relative_features(self):
        # calculate z-scores for game features relative to genre averages.
        self.genre = ParallelGenre(self)

        result = self.features.copy()
        # list of base feature names
        base_features = [
            col
            for col in self.features
            if not col.startswith("audience_overlap_relative_")
        ]
        # calculate z-score for each feature
        for feature in base_features:
            # get the corresponding std dev column name
            std_col = f"{feature}_std"

            # calculate z-score: (x - μ) / σ
            # use safe division to handle zero standard deviations
            z_score = (
                self.features[feature] - self.genre.averages[feature]
            ) / self.genre.averages[std_col].replace(0, float("inf"))

            # replace infinite values (from division by zero) with 0
            # this assumes that when std dev is 0, the feature value equals the mean
            z_score = z_score.replace([float("inf"), float("-inf")], 0)

            # add the z-score as a new feature with prefix 'genre_relative_'
            result[f"genre_relative_{feature}"] = z_score

        self.features = result

    def _calcualte_audience_overlap_relative_features(self):
        # calculate z-scores for game features relative to weighted averages across games with high audeinceOverlap.
        self.audience_overlap = ParallelAudienceOverlap(self)

        result = self.features.copy()
        # list of base feature names
        base_features = [
            col for col in self.features if not col.startswith("genre_relative_")
        ]
        # calculate z-score for each feature
        for feature in base_features:
            # get the corresponding std dev column name
            std_col = f"{feature}_std"

            # calculate z-score: (x - μ) / σ
            # use safe division to handle zero standard deviations
            z_score = (
                self.features[feature] - self.audience_overlap.averages[feature]
            ) / self.audience_overlap.averages[std_col].replace(0, float("inf"))

            # replace infinite values (from division by zero) with 0
            # this assumes that when std dev is 0, the feature value equals the mean
            z_score = z_score.replace([float("inf"), float("-inf")], 0)

            # add the z-score as a new feature with prefix 'genre_relative_'
            result[f"audience_overlap_relative_{feature}"] = z_score

        self.features = result


from genre import ParallelAudienceOverlap, ParallelGenre
