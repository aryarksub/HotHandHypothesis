import pandas as pd
import os
from statsmodels.stats.proportion import proportions_ztest
from probability import hhh_prob_for_fixed_num_prev_shots_made, hhh_prob_for_made_shot_streak

def multi_pairwise_z_tests(probabilities, num_samples, alpha):
    # Treat probabilities as proportions for multiple z-tests using Bonferroni correction
    # For each pair of probabilities, run a z-test for the hypothesis that the difference
    # in probabilities is zero (i.e. the probabilities are equal)

    N = len(probabilities)
    num_tests = N * (N - 1) // 2
    alpha_adjusted = alpha / num_tests # Bonferroni correction

    p_values = []
    significance = []

    for n1 in range(N):
        prob_n1 = probabilities[n1]
        sample_size_n1 = num_samples[n1]

        p_values_n1 = [None] * N
        significance_n1 = [None] * N

        for n2 in range(n1 + 1, N):
            prob_n2 = probabilities[n2]
            sample_size_n2 = num_samples[n2]

            successes = [int(prob_n1 * sample_size_n1), int(prob_n2 * sample_size_n2)]
            trials = [sample_size_n1, sample_size_n2]
            z_score, p_value = proportions_ztest(successes, trials)

            p_values_n1[n2] = p_value
            significance_n1[n2] = (p_value < alpha_adjusted)
        
        p_values.append(p_values_n1)
        significance.append(significance_n1)
    
    return p_values, significance

def run_pairwise_z_tests_k_of_n(probabilities, shot_sample_sizes, diff_adj=False, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)

    for n in range(N):
        num_tests = (n + 1) * (n + 2) // 2

        p_values_n, significance_n = multi_pairwise_z_tests(probabilities[n], shot_sample_sizes[n], alpha)
        sig_indices = [(i, j) for i, row in enumerate(significance_n) for j, sig in enumerate(row) if sig]
        
        if verbose:
            print(f"Assessing {'DA' if diff_adj else ''}HHH for k made in last {n+1} shots: ", end='')
            if sig_indices:
                print(f"k-values for which probability difference is significant: {sig_indices}")
            else:
                print("There is no pair of k-values for which the probability difference is significant")

        if dir_name and file_prefix:
            os.makedirs(dir_name, exist_ok=True)

            width = 7 # spaces for text width

            with open(f"{dir_name}\{file_prefix}_n={n+1}.txt", "w") as file:
                file.write(f"Table of pairwise p-values for conditional prob differences between i,j makes in last {n+1} shots ({'' if diff_adj else 'NOT '}adjusting for difficulty)\n")
                file.write(f"Using Bonferroni-corrected alpha value of {round(alpha / num_tests, 5)}\n\n")
                file.write("i,j".center(width) + " ")
                file.write(" ".join([str(x).center(width) for x in range(n+2)]))
                file.write("\n")
                for k in range(n+2):
                    file.write(str(k).center(width) + " ")
                    file.write(" ".join([str(round(p, 5)).center(width) if p else 'X'.center(width) for p in p_values_n[k]]))
                    file.write("\n")
                if sig_indices:
                    file.write(f"\nk-values for which probability difference is significant: {sig_indices}\n")
                else:
                    file.write("\nThere is no pair of k-values for which the probability difference is significant\n")

def run_pairwise_z_tests_n_straight(probabilities, shot_sample_sizes, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)
    num_tests = (N + 1) * (N + 2) // 2

    p_values, significance = multi_pairwise_z_tests(probabilities, shot_sample_sizes, alpha)
    shot_pairs_with_sig = [(i+1, j+1) for i, row in enumerate(significance) for j, sig in enumerate(row) if sig]

    if verbose:
        print(f"Assessing HHH for increasing streaks of previous n made shots (n = 1 to {N}): ", end='')
        if shot_pairs_with_sig:
            print(f"n-values for which probability difference is significant: {shot_pairs_with_sig}")
        else:
            print("There is no pair of n-values for which the probability difference is significant")

    if dir_name and file_prefix:
        os.makedirs(dir_name, exist_ok=True)

        width = 7 # spaces for text width

        with open(f"{dir_name}\{file_prefix}.txt", "w") as file:
            file.write(f"Table of pairwise p-values for conditional prob differences between streaks of previous i,j made shots\n")
            file.write(f"Using Bonferroni-corrected alpha value of {round(alpha / num_tests, 5)}\n\n")
            file.write("i,j".center(width) + " ")
            file.write(" ".join([str(x).center(width) for x in range(1, N+1)]))
            file.write("\n")
            for k in range(N):
                file.write(str(k+1).center(width) + " ")
                file.write(" ".join([str(round(p, 5)).center(width) if p else 'X'.center(width) for p in p_values[k]]))
                file.write("\n")
            if shot_pairs_with_sig:
                file.write(f"\nn-values for which probability difference is significant: {shot_pairs_with_sig}\n")
            else:
                file.write("\nThere is no pair of n-values for which the probability difference is significant\n")

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10

    probabilities_hhh_k_of_n, shot_sample_sizes_hhh_k_of_n = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, plot=False)
    run_pairwise_z_tests_k_of_n(
        probabilities_hhh_k_of_n, shot_sample_sizes_hhh_k_of_n, 
        diff_adj=False, verbose=False, 
        dir_name="results\hhh_k_of_n", file_prefix="p_value_table"
    )

    probabilities_hhh_n_straight, probabilities_hhh_one_prev_miss, shot_sample_sizes_hhh_n_straight, shot_sample_sizes_hhh_one_prev_miss = hhh_prob_for_made_shot_streak(df, max_shot_memory, plot=True)
    run_pairwise_z_tests_n_straight(
        probabilities_hhh_n_straight, shot_sample_sizes_hhh_n_straight,
        verbose=False,
        dir_name="results\hhh_n_straight", file_prefix="p_value_table"
    )

    probabilities_dahhh_k_of_n_d_gt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_gt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="greater", plot=True)
    run_pairwise_z_tests_k_of_n(
        probabilities_dahhh_k_of_n_d_gt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_gt_d_avg, 
        diff_adj=True, verbose=False, 
        dir_name="results\dahhh_k_of_n\gt", file_prefix="p_value_table"
    )

    probabilities_dahhh_k_of_n_d_lt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_lt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="less", plot=True)
    run_pairwise_z_tests_k_of_n(
        probabilities_dahhh_k_of_n_d_lt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_lt_d_avg, 
        diff_adj=True, verbose=False, 
        dir_name="results\dahhh_k_of_n\lt", file_prefix="p_value_table"
    )