import pandas as pd

def get_prob_make_given_num_shots_made(df, num_prev_shots, num_shots_made):
    df_sub = df.loc[df[f"TOT_FGM-{num_prev_shots}"] == num_shots_made]
    return df_sub["FGM"].sum() / len(df_sub)

def get_prob_make_given_result_sequence(df, num_prev_shots, sequence):
    df_sub = df.loc[df[f"PREV-{num_prev_shots}"] == sequence]
    return df_sub["FGM"].sum() / len(df_sub)

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10
    
    # Probability of making a shot given k makes in last n shots (1 <= n <= 10, 0 <= k <= n)
    for n in range(1, max_shot_memory+1):
        for k in range(0, n+1):
            prob_make_with_k_makes_in_last_n_shots = get_prob_make_given_num_shots_made(df, n, k)
            print(f"Prob make given {k} makes in last {n} shots:", round(prob_make_with_k_makes_in_last_n_shots, 4))

    # Probability of making a shot given n makes in last n shots (1 <= n <= 10)
    for n in range(1, max_shot_memory+1):
        prob_make_with_n_cons_prev_makes = get_prob_make_given_num_shots_made(df, n, n)
        print(f"Prob make given consecutive previous {n} makes:", round(prob_make_with_n_cons_prev_makes, 4))
