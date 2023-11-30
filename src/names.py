import pandas as pd

from data_processor import get_cleaned_name, get_converted_name

def get_mismatched_names(official_names):
    shots_df = pd.read_csv("data/shots.csv")
    shooter_names = {get_cleaned_name(name) for name in shots_df["player_name"].values}
    defender_names = {get_cleaned_name(get_converted_name(name)) for name in shots_df["CLOSEST_DEFENDER"].values} 
    
    height_df = pd.read_csv("data/all_seasons.csv")
    season_stats = height_df[height_df["season"] == "2014-15"]
    player_names = {get_cleaned_name(name) for name in season_stats["player_name"].values}

    names = shooter_names.union(defender_names, player_names)
    return {name for name in names if name not in official_names}

def get_official_names_list():
    adv_stats_df = pd.read_csv("data/advanced_stats.csv")
    cleaned_names = {get_cleaned_name(name) for name in adv_stats_df["Player"].values}
    return cleaned_names

if __name__ == '__main__':
    wrong_names = get_mismatched_names(get_official_names_list())
    for name in wrong_names:
        print(name)