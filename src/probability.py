import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
from scipy.signal import savgol_filter

from models import logistic_regression, xgboost

def make_scatter_plot(x, y_data, line_graph=False, remove_x_ticks=False, xlabel=None, ylabel=None, plot_labels=None, title=None, dir_name=None, file_name=None):
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    for i in range(len(y_data)):
        if line_graph:
            plt.plot(x, y_data[i], label=(plot_labels[i] if plot_labels else None))
        else:
            plt.scatter(x, y_data[i], label=(plot_labels[i] if plot_labels else None))
    
    if remove_x_ticks:
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    else:
        # use 10 evenly spaced ticks in the plot if there are many data points
        indices = np.round(np.linspace(0, len(x) - 1, 10)).astype(int)
        tick_locs = np.round(x[indices], 2) if len(x) > 20 else np.round(x, 2)
        plt.xticks(ticks=tick_locs, labels=tick_locs)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if plot_labels:
        plt.legend(loc='best')
    
    plt.savefig(f"{dir_name}\{file_name}")
    plt.clf()

def generate_shot_sequences(N):
    # Generate all possible shot sequences of length N: (s_1, s_2, ..., s_N)
    # These represent the immediately previous shot, 2nd prev shot, etc.
    # Output is ordered as follows:
    #   1. Number of shots made (e.g. 000 comes before 001, 010, 100)
    #   2. Most recent shot made (e.g. 0011 < 0101 < 0110 < 1001)
    # Sequence defined formally here: https://oeis.org/A294648

    def to_padded_binary(val):
        return bin(val)[2:].zfill(N)

    dp = [[[] for _ in range(N+1)] for _ in range(N+1)]
    for n in range(N+1):
        dp[n][0] = [0]
        dp[n][n] = [2**n - 1]
    for n in range(N+1):
        for k in range(1, n):
            dp[n][k] = dp[n-1][k] + [2**(n-1) + x for x in dp[n-1][k-1]]

    return [to_padded_binary(sequence) for k in range(N+1) for sequence in dp[N][k]]

def get_prob_make_given_num_shots_made(df, num_prev_shots, min_num_shots_made, max_num_shots_made, diff_adj=""):
    # If diff_adj != "", get sub-dataframe based on shots with difficulty above/below prev shot avg difficulty
    if diff_adj == "greater":
        df_diff_adj = df.loc[df["SHOT_DIFFICULTY"] >= df[f"D_AVG-{num_prev_shots}"]]
    elif diff_adj == "lower":
        df_diff_adj = df.loc[df["SHOT_DIFFICULTY"] < df[f"D_AVG-{num_prev_shots}"]]
    else:
        df_diff_adj = df
    df_sub = df_diff_adj.loc[df_diff_adj[f"TOT_FGM-{num_prev_shots}"].between(min_num_shots_made, max_num_shots_made, inclusive='both')]

    # Return probability and number of total shots with given property
    return df_sub["FGM"].sum() / len(df_sub), len(df_sub)

def get_prob_make_given_result_sequence(df, num_prev_shots, sequence):
    # Prepend "X" to stay consistent with data format in shot_result_dataset.csv
    mod_sequence = "X" + sequence
    df_sub = df.loc[df[f"PREV-{num_prev_shots}"] == mod_sequence]
    return df_sub["FGM"].sum() / len(df_sub)

def hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="", verbose=False, plot=False):
    # Set description/plot names to indicate whether prob is calculated using shots above/below prev shot avg difficulty
    diff_description = (f" & D {'>=' if diff_adj == 'greater' else '<'} D_avg") if diff_adj else ""
    plot_dir_name = (f"plots\dahhh_k_of_n\{'gt' if diff_adj == 'greater' else 'lt'}") if diff_adj else "plots\hhh_k_of_n"  
    plot_file_name_suffix = (f"_d_{'gt' if diff_adj == 'greater' else 'lt'}") if diff_adj else ""

    probabilities = []
    shot_sample_sizes = []
    
    # Probability of making a shot given k makes in last n shots (1 <= n <= 10, 0 <= k <= n)
    for n in range(1, max_shot_memory+1):
        prob_for_n_shots = []
        shot_sample_sizes_n = []
        
        for k in range(n+1):
            prob_make_with_k_makes_in_last_n_shots, shot_sample_size = get_prob_make_given_num_shots_made(df, n, k, k, diff_adj=diff_adj)
            prob_for_n_shots.append(prob_make_with_k_makes_in_last_n_shots)
            shot_sample_sizes_n.append(shot_sample_size)
            
            if verbose:
                print(
                    f"Prob make given {k} makes in last {n} shots{diff_description}:", 
                    round(prob_make_with_k_makes_in_last_n_shots, 4)
                )

        if plot:
            make_scatter_plot(
                range(n+1), 
                [prob_for_n_shots], 
                xlabel="k", 
                ylabel="Probability", 
                title=f"P(Make shot n+1 | Make k out of n previous shots{diff_description}): n={n}",
                dir_name=plot_dir_name,
                file_name=f"prob_make_given_k_out_of_{n}_shots{plot_file_name_suffix}.png"
            )
        
        probabilities.append(prob_for_n_shots)
        shot_sample_sizes.append(shot_sample_sizes_n)
    
    return probabilities, shot_sample_sizes

def hhh_prob_for_made_shot_streak(df, max_shot_memory, diff_adj="", verbose=False, plot=False):
    # Set description/plot names to indicate whether prob is calculated using shots above/below prev shot avg difficulty
    diff_description = (f" & D {'>=' if diff_adj == 'greater' else '<'} D_avg") if diff_adj else ""
    plot_dir_name = f"plots\{'da' if diff_adj else ''}hhh_n_straight"
    plot_file_name_suffix = (f"_d_{'gt' if diff_adj == 'greater' else 'lt'}") if diff_adj else ""
    
    probs_n_straight = []
    probs_one_miss = []
    shot_sample_sizes_n_straight = []
    shot_sample_sizes_one_miss = []

    for n in range(1, max_shot_memory+1):
        # Probability of making a shot given n makes in last n shots (1 <= n <= 10)
        prob_make_with_n_cons_prev_makes, shot_sample_size_n_makes = get_prob_make_given_num_shots_made(df, n, n, n, diff_adj=diff_adj)
        # Probability of making a shot given at least 1 miss in last n shots (1 <= n <= 10)
        prob_make_with_at_least_one_prev_miss, shot_sample_size_one_miss = get_prob_make_given_num_shots_made(df, n, 0, n-1, diff_adj=diff_adj)
        probs_n_straight.append(prob_make_with_n_cons_prev_makes)
        probs_one_miss.append(prob_make_with_at_least_one_prev_miss)
        shot_sample_sizes_n_straight.append(shot_sample_size_n_makes)
        shot_sample_sizes_one_miss.append(shot_sample_size_one_miss)

        if verbose:
            print(
                f"Prob make given consecutive previous {n} makes{diff_description}:", 
                round(prob_make_with_n_cons_prev_makes, 4)
            )
            print(
                f"Prob make given at least one miss in previous {n} shots{diff_description}:", 
                round(prob_make_with_at_least_one_prev_miss, 4)
            )
    
    if plot:
        make_scatter_plot(
            range(1, max_shot_memory+1), 
            [probs_n_straight, probs_one_miss], 
            xlabel="n", 
            ylabel="Probability",
            plot_labels=["n straight makes", "1+ misses"],
            title=f"P(Make shot n+1 | Make all previous n shots{diff_description})",
            dir_name=plot_dir_name, 
            file_name=f"prob_make_given_n_straight{plot_file_name_suffix}.png" 
        )

    return probs_n_straight, probs_one_miss, shot_sample_sizes_n_straight, shot_sample_sizes_one_miss

def hhh_prob_for_fixed_length_shot_sequences(df, max_shot_memory, verbose=False, plot=False):
    probabilities = []

    # Probability of making a shot given every shot sequence of length n (1 <= n <= 10)
    for n in range(1, max_shot_memory+1):
        shot_sequences = generate_shot_sequences(n)
        prob_length_n = []
        
        for sequence in shot_sequences:
            prob_make_given_sequence = get_prob_make_given_result_sequence(df, n, sequence)
            prob_length_n.append(prob_make_given_sequence)

            if verbose:
                print(f"Prob make given previous shot result {sequence}:", round(prob_make_given_sequence, 4))

        if plot:
            for k in range(n+1):
                sequences_with_k_makes_indices = [i for i in range(len(shot_sequences)) if shot_sequences[i].count("1") == k]
                probs_for_length_n_seqs_k_makes = [prob_length_n[i] for i in sequences_with_k_makes_indices]
                make_scatter_plot(
                    range(len(sequences_with_k_makes_indices)),
                    [probs_for_length_n_seqs_k_makes],
                    remove_x_ticks=True,
                    xlabel="Ordered shot sequences",
                    ylabel="Probability",
                    title=f"P(Make shot n+1 | Shot sequence of length n, k makes): n={n}, k={k}",
                    dir_name=f"plots\hhh_result_seq\length{n}",
                    file_name=f"prob_make_given_length_{n}_sequence_with_{k}_makes.png"
                )

            make_scatter_plot(
                range(2**n),
                [prob_length_n],
                remove_x_ticks=True,
                xlabel="Ordered shot sequences",
                ylabel="Probability",
                title=f"P(Make shot n+1 | Shot sequence of length n): n={n}",
                dir_name=f"plots\hhh_result_seq\length{n}",
                file_name=f"prob_make_given_length_{n}_sequence_overall.png"
            )
        
        probabilities.append(prob_length_n)
    
    return probabilities

def dahhh_prob_given_diff_adj_shot_metrics(df, metric, max_shot_memory, verbose=False, plot=False):
    probabilities = []

    # metric = "DSS" or "DPTS"
    for n in range(1, max_shot_memory+1):
        metric_col = f"{metric}-{n}"
        in_features = [metric_col]
        df_sub = df.dropna(subset=in_features)

        xgb_clf = xgboost(df_sub, in_features, out_features=["FGM"], verbose=verbose)
        
        max_metric_value = (1 if metric == "DSS" else 3) * n + 0.01 # add extra 0.01 so we can include rightmost endpoint
        max_metric_value = math.ceil(df[metric_col].max()) + 0.01
        metric_values = np.arange(0, max_metric_value, 0.1).reshape(-1, 1)
        prob_make_given_metric = xgb_clf.predict_proba(metric_values)[:, 1]
        probabilities.append(list(zip(metric_values, prob_make_given_metric)))

        if plot:
            # Since probabilities are based on xgb_clf predictions, when plotting the probabilities,
            # we need to plot both with and without lines to preserve consistency across plots
            for plot_with_line in [False, True]:
                make_scatter_plot(
                    np.squeeze(metric_values),
                    [savgol_filter(prob_make_given_metric, 7, 3)] if plot_with_line else [prob_make_given_metric],
                    line_graph=plot_with_line,
                    xlabel=metric,
                    ylabel="Probability",
                    title=f"P(Make shot n+1 | {metric} of prev n shots): n = {n}",
                    dir_name=f"plots\dahhh_{metric}",
                    file_name=f"prob_make_given_{metric}_of_prev_{n}_shots{'_line' if plot_with_line else ''}.png"
                )

    return probabilities

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10

    probabilities_hhh_k_of_n, shot_sample_sizes_hhh_k_of_n = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, plot=True)

    probabilities_hhh_n_straight, probabilities_hhh_one_prev_miss, shot_sample_sizes_hhh_n_straight, shot_sample_sizes_hhh_one_prev_miss = hhh_prob_for_made_shot_streak(df, max_shot_memory, plot=True)

    probabilities_hhh_length_n_sequence = hhh_prob_for_fixed_length_shot_sequences(df, max_shot_memory, plot=True)

    probabilities_dahhh_k_of_n_d_gt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_gt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="greater", plot=True)

    probabilities_dahhh_k_of_n_d_lt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_lt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="less", plot=True)

    for n in range(1, max_shot_memory+1):
        probs_gt, probs_lt = probabilities_dahhh_k_of_n_d_gt_d_avg[n-1], probabilities_dahhh_k_of_n_d_lt_d_avg[n-1]
        make_scatter_plot(
            range(n+1),
            [probs_gt, probs_lt],
            xlabel="k",
            ylabel="Probability",
            plot_labels=["D >= D_AVG", "D < D_AVG"],
            title=f"P(Make shot n+1 | Make k out of n previous shots AND diff adj): n={n}",
            dir_name=f"plots\dahhh_k_of_n\comparison", 
            file_name=f"prob_make_given_k_out_of_{n}_shots_diff_adj.png"
        )

    probabilities_dahhh_n_straight_d_gt, probabilities_dahhh_one_prev_miss_d_gt, shot_sample_sizes_dahhh_n_straight_d_gt, shot_sample_sizes_dahhh_one_prev_miss_d_gt = hhh_prob_for_made_shot_streak(df, max_shot_memory, diff_adj="greater", plot=True)

    probabilities_dahhh_n_straight_d_lt, probabilities_dahhh_one_prev_miss_d_lt, shot_sample_sizes_dahhh_n_straight_d_lt, shot_sample_sizes_dahhh_one_prev_miss_d_lt = hhh_prob_for_made_shot_streak(df, max_shot_memory, diff_adj="less", plot=True)

    make_scatter_plot(
        range(1, max_shot_memory+1),
        [
            probabilities_dahhh_n_straight_d_gt, probabilities_dahhh_n_straight_d_lt,
            probabilities_dahhh_one_prev_miss_d_gt, probabilities_dahhh_one_prev_miss_d_lt
        ],
        xlabel="n",
        ylabel="Probability",
        plot_labels=["n straight D >= D_AVG", "n straight D < D_AVG", "1+ misses D >= D_AVG", "1+ misses D < D_AVG"],
        title="P(Make shot n+1 | Make all previous n shots AND diff adj)",
        dir_name="plots\dahhh_n_straight",
        file_name="prob_make_given_n_straight_diff_adj.png"
    )

    probabilities_dahhh_dss = dahhh_prob_given_diff_adj_shot_metrics(df, "DSS", max_shot_memory, plot=True)

    probabilities_dahhh_dpts = dahhh_prob_given_diff_adj_shot_metrics(df, "DPTS", max_shot_memory, plot=True)