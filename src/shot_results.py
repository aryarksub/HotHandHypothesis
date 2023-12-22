import numpy as np
import pandas as pd

def add_previous_shot_results(df, num_shots=5):
    # Result of shot taken i shots ago (1 <= i <= num_shots)
    for shot in range(1, num_shots+1):
        df[f"FGM-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, df["FGM"].shift(shot), 0).astype(np.int8)
        
    # Num of shots made in last i shots (1 <= i <= num_shots)
    for shot in range(1, num_shots+1):
        df[f"TOT_FGM-{shot}"] = np.where(df["SHOT_NUMBER"] >= shot+1, sum(df[f"FGM-{i}"] for i in range(1, shot+1)), None)

    # Results of last i shots as a string (1 <= i <= num_shots)
    # E.g. "01101" means that shot n-1 missed, n-2 made, n-3 made, n-4 missed, n-5 made
    for shot in range(1, num_shots+1):
        def concatenate_results(row):
            if row["SHOT_NUMBER"] >= shot+1:
                return "".join(str(row[f"FGM-{i}"]) for i in range(1, shot+1))
            else:
                return None

        df[f"PREV-{shot}"] = df.apply(concatenate_results, axis=1) 

    return df

if __name__ == '__main__':
    df = pd.read_csv("data\cleaned_dataset.csv")
    df = df[["GAME_ID", "PLAYER_NAME", "SHOT_NUMBER", "FGM", "SHOT_DIFFICULTY"]]
    df = add_previous_shot_results(df)
    df.to_csv("data\shot_result_dataset.csv", index=False)