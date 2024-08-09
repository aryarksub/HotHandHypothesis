import pandas as pd
import os
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import mannwhitneyu
from probability import hhh_prob_for_fixed_num_prev_shots_made, hhh_prob_for_made_shot_streak, get_fg_and_prob_given_num_shots_made_by_player, dahhh_prob_given_diff_adj_shot_metrics
from statsmodels.tsa.stattools import adfuller
import pymannkendall as pmk

def multi_pairwise_z_tests(probabilities, num_samples, alpha):
    # Treat probabilities as proportions for multiple z-tests using Bonferroni correction
    # For each pair of probabilities, run a two-proportion z-test for the hypothesis that 
    # the difference in probabilities is zero (i.e. the probabilities/proportions are equal)

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
            z_score, p_value = proportions_ztest(successes, trials, alternative='smaller')

            p_values_n1[n2] = p_value
            significance_n1[n2] = (p_value < alpha_adjusted)
        
        p_values.append(p_values_n1)
        significance.append(significance_n1)
    
    return p_values, significance

def multi_pairwise_mann_whitney_tests(probabilities, alpha, bonf_corr=False):
    N = len(probabilities)
    num_tests = N * (N - 1) // 2
    alpha_adjusted = alpha / (num_tests if bonf_corr else 1) # Bonferroni correction

    p_values = []
    significance = []

    for n1 in range(N):
        probs_n1 = probabilities[n1]

        p_values_n1 = [None] * N
        significance_n1 = [None] * N

        for n2 in range(n1 + 1, N):
            probs_n2 = probabilities[n2]

            if len(probs_n1) == 0 or len(probs_n2) == 0:
                p_values_n1[n2] = None
                significance_n1[n2] = None
            else:
                statistic, p_value = mannwhitneyu(probs_n1, probs_n2)
                p_values_n1[n2] = p_value
                significance_n1[n2] = (p_value < alpha_adjusted)

        p_values.append(p_values_n1)
        significance.append(significance_n1)

    return p_values, significance

def multi_augmented_dickey_fuller_tests(probabilities, alpha, metric, verbose=False):
    # probabilities holds N lists, each of which correspond to probability of having certain metric in last n shots
    # Run ADF tests on each of these lists and return p-values and significance/no significance
    # Significance (p-value < alpha) indicates stationarity (no change in statistical properties over increasing metric)
    N = len(probabilities)
    p_values = []
    significance = []

    for n in range(N):
        prob_list_n = probabilities[n]
        adf_test_result = adfuller(prob_list_n)
        p_value = adf_test_result[1]
        sig = p_value < alpha
        p_values.append(p_value)
        significance.append(sig)

        if verbose:
            print(f'Assessing stationarity for DAHHH with {metric} metric using last {n+1} shots: ', end='')
            print(f'p-value = {p_value}\t{"" if sig else "NOT "}STATIONARY')

    return p_values, significance

def multi_mann_kendall_tests(probabilities, alpha, metric, verbose=False):
    # probabilities holds N lists, each of which correspond to probability of having certain metric in last n shots
    # Run ADF tests on each of these lists and return p-values and significance/no significance
    # Significance (p-value < alpha) indicates stationarity (no change in statistical properties over increasing metric)
    N = len(probabilities)
    p_values = []
    trends = []

    for n in range(N):
        prob_list_n = probabilities[n]
        results = pmk.original_test(prob_list_n, alpha=alpha)
        p_value = results.p
        trend = results.trend
        p_values.append(p_value)
        trends.append(trend)

        if verbose:
            print(f'Assessing monotonicity for DAHHH with {metric} metric using last {n+1} shots: ', end='')
            print(f'p-value = {p_value}\tTrend: {trend}')
    
    return p_values, trends


def make_result_file_for_2d_table(dir_name, file_name, i_range, j_range, data, header_text="", footer_text="", text_space=7):
    os.makedirs(dir_name, exist_ok=True)

    with open(f"{dir_name}\{file_name}", "w") as file:
        file.write(header_text)

        file.write("i,j".center(text_space) + " ")
        file.write(" ".join([str(x).center(text_space) for x in j_range]))
        file.write("\n")

        for i in range(len(i_range)):
            k = i_range[i]
            file.write(str(k).center(text_space) + " ")
            file.write(" ".join([str(round(p, 5)).center(text_space) if p else 'X'.center(text_space) for p in data[i]]))
            file.write("\n")

        file.write(footer_text)

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
            header = f"Table of pairwise p-values for conditional prob differences between i,j makes in last {n+1} shots ({'' if diff_adj else 'NOT '}adjusting for difficulty)\n" + \
            f"Using Bonferroni-corrected alpha value of {round(alpha / num_tests, 5)}\n\n"
            footer = f"\nk-values for which probability difference is significant: {sig_indices}\n" if sig_indices else \
            "\nThere is no pair of k-values for which the probability difference is significant\n"
            make_result_file_for_2d_table(
                dir_name, f"{file_prefix}_n={n+1}.txt",
                range(n+2), range(n+2), p_values_n,
                header_text=header, footer_text=footer, text_space=7
            )

def run_pairwise_z_tests_n_straight(probabilities, shot_sample_sizes, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)
    num_tests = (N) * (N - 1) // 2

    p_values, significance = multi_pairwise_z_tests(probabilities, shot_sample_sizes, alpha)
    shot_pairs_with_sig = [(i+1, j+1) for i, row in enumerate(significance) for j, sig in enumerate(row) if sig]

    if verbose:
        print(f"Assessing HHH for increasing streaks of previous n made shots (n = 1 to {N}): ", end='')
        if shot_pairs_with_sig:
            print(f"n-values for which probability difference is significant: {shot_pairs_with_sig}")
        else:
            print("There is no pair of n-values for which the probability difference is significant")

    if dir_name and file_prefix:
        header = f"Table of pairwise p-values for conditional prob differences between streaks of previous i,j made shots\n" + \
        f"Using Bonferroni-corrected alpha value of {round(alpha / num_tests, 5)}\n\n"
        footer = f"\nn-values for which probability difference is significant: {shot_pairs_with_sig}\n" if shot_pairs_with_sig else \
        "\nThere is no pair of n-values for which the probability difference is significant\n"
        make_result_file_for_2d_table(
            dir_name, f"{file_prefix}.txt",
            range(1, N+1), range(1, N+1), p_values,
            header_text=header, footer_text=footer, text_space=7
        )

def run_pairwise_mann_whitney_tests_k_of_n(probabilities, diff_adj=False, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)

    for n in range(N):
        num_tests = (n + 1) * (n + 2) // 2

        p_values_n, significance_n = multi_pairwise_mann_whitney_tests(probabilities[n], alpha, bonf_corr=True)
        sig_indices = [(i, j) for i, row in enumerate(significance_n) for j, sig in enumerate(row) if sig]

        if verbose:
            print(f"Assessing {'DA' if diff_adj else ''}HHH for k made in last {n+1} shots: ", end='')
            if sig_indices:
                print(f"k-values for which probability difference is significant: {sig_indices}")
            else:
                print("There is no pair of k-values for which the probability difference is significant")

        if dir_name and file_prefix:
            header = f"Table of pairwise p-values for conditional prob differences between i,j makes in last {n+1} shots ({'' if diff_adj else 'NOT '}adjusting for difficulty)\n" + \
            f"Using Bonferroni-corrected alpha value of {round(alpha / num_tests, 5)} and Mann-Whitney U test\n\n"
            footer = f"\nk-values for which probability difference is significant: {sig_indices}\n" if sig_indices else \
            "\nThere is no pair of k-values for which the probability difference is significant\n"
            make_result_file_for_2d_table(
                dir_name, f"{file_prefix}_n={n+1}.txt",
                range(n+2), range(n+2), p_values_n,
                header_text=header, footer_text=footer, text_space=7
            )

def run_aug_dickey_fuller_tests(probabilities, metric, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)

    clean_probs = [[tup[1] for tup in prob_list] for prob_list in probabilities]

    p_values, significance = multi_augmented_dickey_fuller_tests(clean_probs, alpha, metric, verbose)
    
    if dir_name and file_prefix:
        os.makedirs(dir_name, exist_ok=True)
        text_space = 7

        with open(f"{dir_name}\{file_prefix}", "w") as file:
            file.write(f'Results of Augmented Dickey-Fuller tests run on DAHHH using {metric} metric using last n shots\n\n')

            for n in range(N):
                header = f"n = {n+1}".ljust(text_space) + ": "
                p_val_str = str(round(p_values[n], 5)).ljust(text_space)
                sig_str = ('' if significance[n] else 'NOT ') + 'STATIONARY'
                file.write(f"{header}{p_val_str}\t{sig_str}\n")

def run_mann_kendall_tests(probabilities, metric, verbose=False, dir_name=None, file_prefix=None):
    alpha = 0.05
    N = len(probabilities)

    clean_probs = [[tup[1] for tup in prob_list] for prob_list in probabilities]
    
    p_values, trends = multi_mann_kendall_tests(clean_probs, alpha, metric, verbose)

    if dir_name and file_prefix:
        os.makedirs(dir_name, exist_ok=True)
        text_space = 7

        with open(f"{dir_name}\{file_prefix}", "w") as file:
            file.write(f'Results of Mann-Kendall tests run on DAHHH using {metric} metric using last n shots\n\n')

            for n in range(N):
                header = f"n = {n+1}".ljust(text_space) + ": "
                p_val_str = str(round(p_values[n], 5)).ljust(text_space)
                trend_str = f'Trend: {trends[n]}'
                file.write(f"{header}{p_val_str}\t{trend_str}\n")

def get_probabilities_from_player_metrics_k_of_n(metrics, n):
    probabilities = []
    for n in range(1, n + 1):
        probabilities_n = []

        for k in range(n + 1):
            metric_dict = metrics[n][k]
            probabilities_n_k = [metric_dict[player][0] for player in metric_dict]
            probabilities_n.append(probabilities_n_k)

        probabilities.append(probabilities_n)

    return probabilities

if __name__ == '__main__':
    df = pd.read_csv("data\shot_result_dataset.csv")
    max_shot_memory = 10

    probabilities_hhh_k_of_n, shot_sample_sizes_hhh_k_of_n = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, plot=False)
    run_pairwise_z_tests_k_of_n(
        probabilities_hhh_k_of_n, shot_sample_sizes_hhh_k_of_n, 
        diff_adj=False, verbose=False, 
        dir_name="results\hhh_k_of_n\prop_z", file_prefix="p_value_table"
    )

    metrics_hhh_k_of_n = get_fg_and_prob_given_num_shots_made_by_player(df, max_shot_memory, diff_adj="")
    player_probs_hhh_k_of_n = get_probabilities_from_player_metrics_k_of_n(metrics_hhh_k_of_n, max_shot_memory)
    run_pairwise_mann_whitney_tests_k_of_n(
        player_probs_hhh_k_of_n,
        diff_adj=False, verbose=False,
        dir_name="results\hhh_k_of_n\mann_whitney", file_prefix="p_value_table"
    )

    probabilities_hhh_n_straight, probabilities_hhh_one_prev_miss, shot_sample_sizes_hhh_n_straight, shot_sample_sizes_hhh_one_prev_miss = hhh_prob_for_made_shot_streak(df, max_shot_memory, plot=False)
    run_pairwise_z_tests_n_straight(
        probabilities_hhh_n_straight, shot_sample_sizes_hhh_n_straight,
        verbose=False,
        dir_name="results\hhh_n_straight\prop_z", file_prefix="p_value_table"
    )

    probabilities_dahhh_k_of_n_d_gt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_gt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="greater", plot=False)
    run_pairwise_z_tests_k_of_n(
        probabilities_dahhh_k_of_n_d_gt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_gt_d_avg, 
        diff_adj=True, verbose=False, 
        dir_name="results\dahhh_k_of_n\gt\prop_z", file_prefix="p_value_table"
    )

    probabilities_dahhh_k_of_n_d_lt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_lt_d_avg = hhh_prob_for_fixed_num_prev_shots_made(df, max_shot_memory, diff_adj="less", plot=False)
    run_pairwise_z_tests_k_of_n(
        probabilities_dahhh_k_of_n_d_lt_d_avg, shot_sample_sizes_dahhh_k_of_n_d_lt_d_avg, 
        diff_adj=True, verbose=False, 
        dir_name="results\dahhh_k_of_n\lt\prop_z", file_prefix="p_value_table"
    )

    probabilities_dahhh_dpts = dahhh_prob_given_diff_adj_shot_metrics(df, 'DPTS', max_shot_memory)
    run_aug_dickey_fuller_tests(
        probabilities_dahhh_dpts, 'DPTS', verbose=False, 
        dir_name="results\dahhh_dpts", file_prefix="dpts_adf_p_vals.txt"
    )

    run_mann_kendall_tests(
        probabilities_dahhh_dpts, 'DPTS', verbose=False,
        dir_name="results\dahhh_dpts", file_prefix="dpts_mk_p_vals.txt"
    )

    probabilities_dahhh_dss = dahhh_prob_given_diff_adj_shot_metrics(df, 'DSS', max_shot_memory)
    run_aug_dickey_fuller_tests(
        probabilities_dahhh_dss, 'DSS', verbose=True,
        dir_name="results\dahhh_dss", file_prefix="dss_adf_p_vals.txt"
    )
    
    run_mann_kendall_tests(
        probabilities_dahhh_dss, 'DSS', verbose=False,
        dir_name="results\dahhh_dss", file_prefix="dss_mk_p_vals.txt"
    )