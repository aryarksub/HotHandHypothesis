import matplotlib.pyplot as plt
import os
import pandas as pd

def make_scatter_plot(x, y_data, xlabel=None, ylabel=None, plot_labels=None, title=None, dir_name=None, file_name=None):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    for i in range(len(y_data)):
        plt.scatter(x, y_data[i], label=(plot_labels[i] if plot_labels else None))
    plt.xticks(x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if plot_labels:
        plt.legend(loc='best')
    plt.savefig(f"{dir_name}\{file_name}")
    plt.clf()

def get_prob_make_given_num_shots_made(df, num_prev_shots, min_num_shots_made, max_num_shots_made):
    df_sub = df.loc[df[f"TOT_FGM-{num_prev_shots}"].between(min_num_shots_made, max_num_shots_made, inclusive='both')]
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
            prob_make_with_k_makes_in_last_n_shots = get_prob_make_given_num_shots_made(df, n, k, k)
            prob_for_n_shots.append(prob_make_with_k_makes_in_last_n_shots)
            
            if verbose:
                print(f"Prob make given {k} makes in last {n} shots:", round(prob_make_with_k_makes_in_last_n_shots, 4))

        if plot:
            make_scatter_plot(
                range(n+1), 
                [prob_for_n_shots], 
                xlabel="k", 
                ylabel="Probability", 
                title=f"P(Make shot n+1 | Make k out of n previous shots): n={n}",
                dir_name="plots\hhh_k_of_n", 
                file_name=f"prob_make_given_k_out_of_{n}_shots.png"
            )
        
        probabilities.append(prob_for_n_shots)
    
    return probabilities

def hhh_prob_for_made_shot_streak(df, max_shot_memory, verbose=False, plot=False):
    probs_n_straight = []
    probs_one_miss = []

    for n in range(1, max_shot_memory+1):
        # Probability of making a shot given n makes in last n shots (1 <= n <= 10)
        prob_make_with_n_cons_prev_makes = get_prob_make_given_num_shots_made(df, n, n, n)
        # Probability of making a shot given at least 1 miss in last n shots (1 <= n <= 10)
        prob_make_with_at_least_one_prev_miss = get_prob_make_given_num_shots_made(df, n, 0, n-1)
        probs_n_straight.append(prob_make_with_n_cons_prev_makes)
        probs_one_miss.append(prob_make_with_at_least_one_prev_miss)

        if verbose:
            print(f"Prob make given consecutive previous {n} makes:", round(prob_make_with_n_cons_prev_makes, 4))
            print(f"Prob make given at least one miss in previous {n} shots:", round(prob_make_with_at_least_one_prev_miss, 4))
    
    if plot:
        make_scatter_plot(
            range(1, max_shot_memory+1), 
            [probs_n_straight, probs_one_miss], 
            xlabel="n", 
            ylabel="Probability",
            plot_labels=["n straight makes", "1+ misses"],
            title="P(Make shot n+1 | Make all previous n shots)",
            dir_name="plots\hhh_n_straight", 
            file_name="prob_make_given_n_straight.png" 
        )

    return probs_n_straight, probs_one_miss

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10

    probabilities_hhh_k_of_n = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, plot=True)

    probabilities_hhh_n_straight, probabilities_hhh_one_prev_miss = hhh_prob_for_made_shot_streak(df, max_shot_memory, plot=True)
