import numpy as np
import requests
import os


def game_get_request(id):
    # call gamealytic api for given game id
    headers = {
        "api-key": f"{os.environ.get("SECRET_KEY")}",
        "Content-Type": "application/json",
    }
    url = f"https://api.gamalytic.com/game/{id}?fields=name%2Cgenres%2Ctags%2CfirstReleaseDate%2Chistory&include_pre_release_history=true"
    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def publisher_games_get_request(id):
    # call gamealytic api for given publisher id
    headers = {
        "api-key": f"{os.environ.get("SECRET_KEY")}",
        "Content-Type": "application/json",
    }
    url = f"https://api.gamalytic.com/publishers/{id}"
    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def similar_games_get_request(genres, tags, release_date, broad=False):
    # call gamealytic api for list of similar game ids by genre, tags and release date
    headers = {
        "api-key": f"{os.environ.get("SECRET_KEY")}",
        "Content-Type": "application/json",
    }
    # convert list to url string
    genres_string = ",".join(genres)
    genres_url = genres_string.replace(" ", "%20").replace(",", "%2C")
    tags_string = ",".join(tags)
    tags_url = tags_string.replace(" ", "%20").replace(",", "%2C")

    # range for games to consider
    one_year = 31556926000
    date_min = release_date - one_year
    date_max = release_date + one_year
    if broad:
        date_min = release_date - 2 * one_year
        date_max = release_date + 2 * one_year

    url = f"https://api.gamalytic.com/steam-games/list?page=0&limit=100&fields=id&sort=copiesSold&sort_mode=desc&genres={genres_url}&tags={tags_url}&first_release_date_min={date_min}&first_release_date_max={date_max}"
    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None


def audience_overlap_games_get_request(id):
    # call gamealytic api for list of similar game ids by genre, tags and release date
    headers = {
        "api-key": f"{os.environ.get("SECRET_KEY")}",
        "Content-Type": "application/json",
    }

    url = f"https://api.gamalytic.com/game/{id}?fields=audienceOverlap"
    try:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            posts = response.json()
            return posts
        else:
            print("Error:", response.status_code)
            return None
    except requests.exceptions.RequestException as e:
        print("Error:", e)
        return None
