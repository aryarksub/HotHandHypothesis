import numpy as np
import pandas as pd

def add_previous_shot_results(df, num_shots=5):
    for shot in range(1, num_shots+1):
        # Result of shot taken i shots ago (1 <= i <= num_shots)
        df[f"FGM-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, df["FGM"].shift(shot), 0).astype(np.int8)
        
        # Num of shots made in last i shots (1 <= i <= num_shots)
        df[f"TOT_FGM-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, sum(df[f"FGM-{i}"] for i in range(1, shot+1)), None)

        # Results of last i shots as a string (1 <= i <= num_shots)
        # E.g. "01101" means that shot n-1 missed, n-2 made, n-3 made, n-4 missed, n-5 made
        def concatenate_results(row):
            if row["SHOT_NUMBER"] >= shot+1:
                return "X" + "".join(str(row[f"FGM-{i}"]) for i in range(1, shot+1))
            else:
                return None

        df[f"PREV-{shot}"] = df.apply(concatenate_results, axis=1) 

    return df

def add_previous_shot_difficulty_metrics(df, num_shots=5):
    for shot in range(1, num_shots+1):
        # Avg difficulty of last i shots (1 <= i <= num_shots)
        df[f"D_AVG-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, sum(df["SHOT_DIFFICULTY"].shift(i) for i in range(1, shot+1)) / shot, None)

        # Difficulty-weighted shot success of last i shots (1 <= i <= num_shots): sum_{j = 1 to i} (s_{n-j} * d_{n-j})
        df[f"DSS-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, sum(df["SHOT_DIFFICULTY"].shift(i) * df["FGM"].shift(i) for i in range(1, shot+1)), None)

        # Difficulty-weighted points made of last i shots (1 <= i <= num_shots): sum_{j = 1 to j} (d_{n-j} * p_{n-j} * s_{n-j})
        df[f"DPTS-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, sum(df["SHOT_DIFFICULTY"].shift(i) * df["PTS_TYPE"].shift(i) * df["FGM"].shift(i) for i in range(1, shot+1)), None)

    return df

if __name__ == '__main__':
    df = pd.read_csv("data\cleaned_dataset.csv")
    df = df[["GAME_ID", "PLAYER_NAME", "SHOT_NUMBER", "PTS_TYPE", "FGM", "SHOT_DIFFICULTY"]]
    df = add_previous_shot_results(df, num_shots=10)
    df = add_previous_shot_difficulty_metrics(df, num_shots=10)
    df.to_csv("data\shot_result_dataset.csv", index=False)