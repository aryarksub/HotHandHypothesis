import matplotlib.pyplot as plt
import os
import pandas as pd

def make_scatter_plot(x, y, xlabel=None, ylabel=None, title=None, dir_name=None, file_name=None):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    plt.scatter(x, y)
    plt.xticks(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(f"{dir_name}\{file_name}")
    plt.clf()

def get_prob_make_given_num_shots_made(df, num_prev_shots, num_shots_made):
    df_sub = df.loc[df[f"TOT_FGM-{num_prev_shots}"] == num_shots_made]
    return df_sub["FGM"].sum() / len(df_sub)

def get_prob_make_given_result_sequence(df, num_prev_shots, sequence):
    df_sub = df.loc[df[f"PREV-{num_prev_shots}"] == sequence]
    return df_sub["FGM"].sum() / len(df_sub)

def hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, verbose=False, plot=False):
    probabilities = []
    
    # Probability of making a shot given k makes in last n shots (1 <= n <= 10, 0 <= k <= n)
    for n in range(1, max_shot_memory+1):
        prob_for_n_shots = []
        
        for k in range(n+1):
            prob_make_with_k_makes_in_last_n_shots = get_prob_make_given_num_shots_made(df, n, k)
            prob_for_n_shots.append(prob_make_with_k_makes_in_last_n_shots)
            
            if verbose:
                print(f"Prob make given {k} makes in last {n} shots:", round(prob_make_with_k_makes_in_last_n_shots, 4))

        if plot:
            make_scatter_plot(
                range(n+1), prob_for_n_shots, "k", "Probability", 
                f"P(Make shot n+1 | Make k out of n previous shots): n={n}",
                "plots\hhh_k_of_n", f"prob_make_given_k_out_of_{n}_shots.png"
            )
        
        probabilities.append(prob_for_n_shots)
    
    return probabilities

def hhh_prob_for_made_shot_streak(df, max_shot_memory, verbose=False, plot=False):
    probabilities = []

    # Probability of making a shot given n makes in last n shots (1 <= n <= 10)
    for n in range(1, max_shot_memory+1):
        prob_make_with_n_cons_prev_makes = get_prob_make_given_num_shots_made(df, n, n)
        probabilities.append(prob_make_with_n_cons_prev_makes)

        if verbose:
            print(f"Prob make given consecutive previous {n} makes:", round(prob_make_with_n_cons_prev_makes, 4))
    
    if plot:
        make_scatter_plot(
            range(1, max_shot_memory+1), probabilities, "n", "Probability",
            "P(Make shot n+1 | Make all previous n shots)",
            "plots\hhh_n_straight", "prob_make_given_n_straight.png" 
        )

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10

    probabilities_hhh_k_of_n = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, plot=True)

    probabilities_hhh_n_straight = hhh_prob_for_made_shot_streak(df, max_shot_memory, plot=True)
