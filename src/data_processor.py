import pandas as pd

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

    return df