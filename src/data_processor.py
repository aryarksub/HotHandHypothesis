import pandas as pd
from unidecode import unidecode
from xgboost import XGBClassifier

from simple_reg import add_all_regression_columns, add_all_regression_binned_columns

name_replacements = {
    "charles hayes" : "chuck hayes",
    "jon ingles" : "joe ingles",
    "perry jones iii" : "perry jones",
    "jose juan barea" : "jj barea",
    "otto porter" : "otto porter jr",
    "jimmer dredette" : "jimmer fredette",
    "glenn robinson" : "glenn robinson iii",
    "amare stoudemire" : "amar'e stoudemire",
    "danilo gallinai" : "danilo gallinari",
    "kyle oquinn" : "kyle o'quinn",
    "time hardaway jr" : "tim hardaway jr",
    "nene hilario" : "nene",
    "james ennis" : "james ennis iii",
    "mnta ellis" : "monta ellis",
    "enes kanter" : "enes freedom",
    "toure murry" : "toure' murry",
    "dwayne wade" : "dwyane wade",
    "glen rice" : "glen rice jr",
    "steve adams" : "steven adams",
    "al farouq aminu" : "al-farouq aminu",
    "johnny o'bryant iii" : "johnny o'bryant",
    "dirk nowtizski" : "dirk nowitzki",
    "alan crabbe" : "allen crabbe",
    "jeffery taylor" : "jeff taylor",
    "beno urdih" : "beno udrih",
    "nerles noel" : "nerlens noel",
    "larry drew" : "larry drew ii"
}

def get_cleaned_name(name):
    return ''.join(filter(lambda x: x.isalpha() or x in " '-", unidecode(name))).lower()

def replace_names(df):
    return df.replace(name_replacements)

def get_converted_name(name):
    # Convert "Last, First" to "first last" (lowercase)
    tokens = [token.strip().lower() for token in name.split(",")]
    new_name = " ".join(tokens[::-1])
    return get_cleaned_name(new_name)

def get_player_to_height_dict():
    all_stats = pd.read_csv("data/all_seasons.csv")
    season_stats = all_stats[all_stats["season"] == "2014-15"]
    names = [get_cleaned_name(name) for name in season_stats["player_name"].values]
    replaced_names = [name_replacements.get(name, name) for name in names]
    height_dict = dict(zip(replaced_names, season_stats.player_height))
    return height_dict

def add_heights(df):
    heights = get_player_to_height_dict()

    df["HEIGHT"] = df.apply(lambda row : heights[row["PLAYER_NAME"]], axis=1)
    df["HEIGHT_DIFF"] = df.apply(
        lambda row : heights[row["PLAYER_NAME"]] - heights[get_converted_name(row["CLOSEST_DEFENDER"])], axis=1
    )
    return df

def add_off_def_rtg(df):
    adv_stats = pd.read_csv("data/advanced_stats.csv")
    names = [get_cleaned_name(name) for name in adv_stats["Player"]]
    off_def_rtg = zip(adv_stats["ORtg"], adv_stats["DRtg"])
    rtg_dict = dict(zip(names, off_def_rtg))

    df["OFF_RTG"] = df.apply(lambda row : rtg_dict[row["PLAYER_NAME"]][0], axis=1)
    df["DEF_RTG"] = df.apply(lambda row : rtg_dict[row["PLAYER_NAME"]][1], axis=1)
    df["RTG_DIFF"] = df.apply(lambda row : row["OFF_RTG"] - row["DEF_RTG"], axis=1)
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
    bin_data("CLOSE_DEF_DIST_BIN", "CLOSE_DEF_DIST", [0, 2, 4, 6, 100])
    return df

def add_shot_difficulty(df):
    def construct_optimal_xgb_model(df, in_features, out_features):
        extra_params = {'n_estimators': 104, 'reg_alpha': 0.5121225370235516, 'reg_lambda': 1.8195910691785828, 'subsample': 0.7}
        opt_model = XGBClassifier(max_depth=3, max_leaves=10, learning_rate=0.14503807928342588, **extra_params)
        fit_model = opt_model.fit(df[in_features], df[out_features])
        return fit_model
    
    in_features = ['DRIBBLES_POLY', 'SHOT_DIST', 'CLOSE_DEF_DIST_EXP', 'HEIGHT_DIFF', 'PERIOD', 'SHOT_CLOCK', 'HOME']
    out_features = ['FGM']
    difficulty_model = construct_optimal_xgb_model(df, in_features, out_features)
    df["SHOT_DIFFICULTY"] = difficulty_model.predict_proba(df[in_features])[:,0]
    return df

def get_cleaned_shot_data(csv_name=None):
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

    # Atila dos Santos does not have relevant data in other datasets, so we remove corresponding rows from analysis dataframe
    df = df[(df["PLAYER_NAME"] != "atila dos santos") & (df["CLOSEST_DEFENDER"] != "Dos Santos, Atila")]

    df["PLAYER_NAME"] = df["PLAYER_NAME"].apply(get_cleaned_name)
    df["CLOSEST_DEFENDER"] = df["CLOSEST_DEFENDER"].apply(get_converted_name)

    df = replace_names(df)
    df = add_heights(df)
    df = add_off_def_rtg(df)
    df = add_binned_data(df)
    df = add_all_regression_columns(df)
    df = add_all_regression_binned_columns(df)
    df = add_shot_difficulty(df)

    if csv_name:
        df.to_csv(csv_name, index=False)

    return df

if __name__ == '__main__':
    df = get_cleaned_shot_data("data/cleaned_dataset.csv")
    print(df.head())