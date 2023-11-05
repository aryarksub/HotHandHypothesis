import numpy as np
import scipy.optimize

def r_sq(val, pred):
    sqDiff = np.square(val - pred)
    sqDiffFromMean = np.square(val - np.mean(val))
    return 1 - np.sum(sqDiff) / np.sum(sqDiffFromMean)

def exp_fit(x, a, b, c):
    return a * np.exp(b*x) + c

def poly_fit(x, y, degree):
    return np.polyfit(x, y, degree)

def get_shot_pct_by_intervals(df, column, intervals):
    shot_pct_by_dribble = []
    for interval in intervals:
        low, high = interval
        df_sub = df[(low <= df[column]) & (df[column] <= high)]
        made = df_sub["FGM"].sum()
        missed = df_sub["FGM"].shape[0] - made
        shot_pct_by_dribble.append(made / (made + missed))
    return shot_pct_by_dribble

def add_exp_reg_column(df, column, intervals, init_params, verbose=False):
    if verbose:
        print(f"Adding exponential regression column for {column}")
    shot_pct = get_shot_pct_by_intervals(df, column, intervals)
    x_range = [sum(interval) / 2 for interval in intervals]

    params, _ = scipy.optimize.curve_fit(exp_fit, x_range, shot_pct, init_params)
    a, b, c = params
    if verbose:
        print(f"Model params [a,b,c] for y = a * e^(bx) + c: {params}")
        print(f"R Squared: {r_sq(shot_pct, exp_fit(np.asarray(x_range, dtype=np.float64), a, b, c))}")

    df[f"{column}_EXP"] = exp_fit(df[column].astype(float), a, b, c)
    
    return df

def add_poly_reg_column(df, column, intervals, degree, verbose=False):
    if verbose:
        print(f"Adding polynomial regression (degree={degree}) column for {column}")
    shot_pct = get_shot_pct_by_intervals(df, column, intervals)
    x_range = [sum(interval) / 2 for interval in intervals]

    poly_model_params = poly_fit(x_range, shot_pct, degree)
    if verbose:
        print(f"Model params (coeff for highest degree first): {poly_model_params}")
        print(f"R Squared: {r_sq(shot_pct, np.poly1d(poly_model_params)(x_range))}")

    df[f"{column}_POLY"] = np.poly1d(poly_model_params)(df[column])
    
    return df

def add_all_regression_columns(df, verbose=False):
    # Regression based on dribbles
    # intervals determined by looking at number of data points for each number of dribbles
    # and splitting into bins with approximately same number of data points
    dribbles_intervals = [(0,0), (1,1), (2,2), (3,4), (5,9), (10,17)]

    df = add_exp_reg_column(df, "DRIBBLES", dribbles_intervals, (0, -1, 0), verbose)
    df = add_poly_reg_column(df, "DRIBBLES", dribbles_intervals, 3, verbose)

    # Regression based on shot distance
    epsilon = 0.01 # use some small value to ensure intervals don't overlap 
    bin_length = 4
    shot_dist_intervals = [(x + epsilon, x + bin_length) for x in range(0, 33, bin_length)]

    df = add_poly_reg_column(df, "SHOT_DIST", shot_dist_intervals, 3, verbose)

    # Regression based on closest defender distance
    bin_length = 2
    def_dist_intervals = [(x + epsilon, x + bin_length) for x in range(0, 30, bin_length)]

    df = add_exp_reg_column(df, "CLOSE_DEF_DIST", def_dist_intervals, (0, 0, 0), verbose)
    df = add_poly_reg_column(df, "CLOSE_DEF_DIST", def_dist_intervals, 2, verbose)

    return df

def add_all_regression_binned_columns(df, verbose=False):
    # Regression based on binned dribbles
    dribbles_bin_intervals = [(i,i) for i in range(df["DRIBBLES_BIN"].max()+1)]
    df = add_exp_reg_column(df, "DRIBBLES_BIN", dribbles_bin_intervals, (0, -1, 0), verbose)
    
    # Regression based on binned shot distance
    shot_dist_bin_intervals = [(i,i) for i in range(df["SHOT_DIST_BIN"].max()+1)]
    df = add_poly_reg_column(df, "SHOT_DIST_BIN", shot_dist_bin_intervals, 3, verbose)

    # There is no good simple regression model for binned closest defender distance data

    return df
