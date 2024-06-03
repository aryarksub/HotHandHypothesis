import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from models import xgb_model
from data_processor import get_cleaned_shot_data

def best_params_for_model(model, param_spaces, df, in_features, out_features):
    # See https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf for random search
    opt_model = RandomizedSearchCV(model, param_spaces, n_iter=50, cv=5, verbose=3)
    fit_model = opt_model.fit(df[in_features], df[out_features])
    return fit_model.best_params_, fit_model.best_score_

if __name__ == '__main__':
    df = get_cleaned_shot_data()

    basic_xgb_model = xgb_model()
    param_space_dict = {
        "n_estimators" : range(100, 151),
        "learning_rate" : stats.uniform(loc=0.05, scale=0.1),
        "max_depth" : [3,4,5],
        "max_leaves" : range(4,11),
        "subsample" : [0.6 + 0.1*x for x in range(5)],
        "reg_alpha" : stats.uniform(loc=0.5, scale=0.25),
        "reg_lambda" : stats.uniform(loc=1.5, scale=0.5)
    }
    # from output of best_features_for_model in model_eval + added features (period, shot clock, home)
    in_features = ['DRIBBLES_POLY', 'SHOT_DIST', 'CLOSE_DEF_DIST_EXP', 'HEIGHT_DIFF', 'PERIOD', 'SHOT_CLOCK', 'HOME']
    out_features = ['FGM']
    best_params, best_score = best_params_for_model(basic_xgb_model, param_space_dict, df, in_features, out_features)
    print(f"Best params for XGBoost: {best_params}\nCorresponding score: {best_score}")