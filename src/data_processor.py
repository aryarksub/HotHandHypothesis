import pandas as pd

name_replacements = {
        "jose juan barea" : "j.j. barea",
        "danilo gallinai" : "danilo gallinari",
        "jimmer dredette" : "jimmer fredette",
        "beno urdih" : "beno udrih",
        "al farouq aminu" : "al-farouq aminu",
        "dirk nowtizski" : "dirk nowitzki",
        "otto porter" : "otto porter jr.",
        "oj mayo" : "o.j. mayo",
        "james ennis" : "james ennis iii",
        "nerles noel" : "nerlens noel",
        "jon ingles" : "joe ingles",
        "amare stoudemire" : "amar'e stoudemire",
        "dj augustin" : "d.j. augustin",
        "nene hilario" : "nene",
        "dwayne wade" : "dwyane wade",
        "kyle oquinn" : "kyle o'quinn",
        "cj watson" : "c.j. watson",
        "time hardaway jr" : "tim hardaway jr.",
        "mnta ellis" : "monta ellis",
        "steve adams" : "steven adams",
        "alan crabbe" : "allen crabbe",
        "charles hayes" : "chuck hayes",
        "johnny o'bryant" : "johnny o'bryant iii",
        "a.j. price" : "aj price",
        "toure murry" : "toure' murry",
        "larry drew" : "larry drew ii",
        "jeff taylor" : "jeffery taylor",
        "j.r. smith" : "jr smith",
        "glenn robinson" : "glenn robinson iii",
        "glen rice jr." : "glen rice",
        "perry jones" : "perry jones iii"
    }

def get_player_to_height_dict():
    all_stats = pd.read_csv("data/all_seasons.csv")
    season_stats = all_stats[all_stats["season"] == "2014-15"]
    names = [name.lower() for name in season_stats["player_name"].values]
    height_dict = dict(zip(names, season_stats.player_height))
    
    # Manual entry since name is not in dataset
    # name -> height in cm
    height_dict["atila dos santos"] = 208.28 # 6'10"
    
    return height_dict

def fix_names(df):
    return df.replace(name_replacements)

def add_heights(df):
    heights = get_player_to_height_dict()

    df["HEIGHT"] = df.apply(lambda row : heights[row["PLAYER_NAME"]], axis=1)
    df["HEIGHT_DIFF"] = df.apply(
        lambda row : heights[row["PLAYER_NAME"]] - heights[get_converted_name(row["CLOSEST_DEFENDER"])], axis=1
    )
    return df

def add_binned_data(df):
    def bin_data(binned_col, orig_col, bins_arr):
        df[binned_col] = pd.cut(
            df[orig_col],
            bins=bins_arr,
            right=False,
            labels=range(len(bins_arr) - 1)
        )

    bin_data("DRIBBLES_BIN", "DRIBBLES", [0, 1, 3, 6, 100])
    bin_data("SHOT_DIST_BIN", "SHOT_DIST", [0,4,8,12,16,24,32,100])
    bin_data("DEF_DIST_BIN", "CLOSE_DEF_DIST", [0, 2, 4, 6, 100])
    return df

def get_cleaned_shot_data():
    def time_to_seconds(time_str):
        mins, secs = time_str.split(":")
        return 60*int(mins) + int(secs)

    df = pd.read_csv("data/shots.csv")

    # Change game clock column from string to integer number of seconds remaining
    df["GAME_CLOCK"] = df["GAME_CLOCK"].map(time_to_seconds)

    # Replace missing shot clock values with game clock value
    df["SHOT_CLOCK"].fillna(df.GAME_CLOCK, inplace=True)

    def get_total_time_left(row):
        # Time left in fourth quarter or OTs is same as time left on game clock
        if row["PERIOD"] >= 4:
            return row["GAME_CLOCK"]
        # 12 * 60 seconds in (4 - period) remaining quarters + game_clock seconds in the current quarter
        return 12 * 60 * (4 - row["PERIOD"]) + row["GAME_CLOCK"]

    df["TOT_TIME_LEFT"] = df.apply(get_total_time_left, axis=1)

    df["HOME"] = df.apply(lambda row : 1 if row["LOCATION"] == "H" else 0, axis=1)

    # Drop columns that aren't important for determining shot difficulty or for hot-hand hypothesis
    df.drop(columns=["MATCHUP", "W", "FINAL_MARGIN", "TOUCH_TIME", "PTS_TYPE", "SHOT_RESULT", "PTS"], inplace=True)

    # Rename columns so all have capitalized names
    df.rename(columns={"player_name" : "PLAYER_NAME", "player_id" : "PLAYER_ID"}, inplace=True)

    df = fix_names(df)
    df = add_heights(df)
    df = add_binned_data(df)

    return df

def get_converted_name(name):
    # Convert "Last, First" to "first last" (lowercase)
    tokens = [token.strip().lower() for token in name.split(",")]
    new_name = " ".join(tokens[::-1])
    return name_replacements.get(new_name, new_name)

def get_wrong_names():
    df = get_cleaned_shot_data()
    all_seasons_stats = pd.read_csv("data/all_seasons.csv")
    season_2014_stats = all_seasons_stats[all_seasons_stats["season"] == "2014-15"]

    player_names = {name.lower() for name in season_2014_stats["player_name"].values}
    wrong_names = set()
    for name in df["PLAYER_NAME"].values:
        if name not in player_names and name not in wrong_names:
            wrong_names.add(name)
    for name in df["CLOSEST_DEFENDER"].values:
        converted_name = get_converted_name(name)
        if converted_name not in player_names and converted_name not in wrong_names:
            wrong_names.add(converted_name)
    return wrong_names

## Uncomment to see set of wrong names
# for name in get_wrong_names():
#     print(name)

## Uncomment to see first few rows of cleaned data
# df = get_cleaned_shot_data()
# print(df.head())