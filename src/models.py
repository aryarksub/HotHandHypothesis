from itertools import combinations
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

from data_processor import get_cleaned_shot_data

def select_in_features(model, df, all_in_features, out_features, max_in_features=None, verbose=False):
    best_score = (0, [])
    kfold_cv = KFold(n_splits=10)

    out_values = df[out_features].values.ravel()

    if max_in_features is None:
        max_in_features = len(all_in_features)

    for num_features in range(1, max_in_features+1):
        all_in_feature_combinations = combinations(all_in_features, num_features)
        for in_features in all_in_feature_combinations:
            in_features = list(in_features)
            scores = cross_val_score(model, df[in_features].values, out_values, cv=kfold_cv)
            accuracy = np.mean(scores)
            if accuracy > best_score[0]:
                best_score = (accuracy, in_features)
        if verbose:
            print(f"Done with all combinations of {num_features} input features")
            print(f"Current best: {best_score}")
    return best_score

def log_model(log_reg_solver='lbfgs', max_iter=100):
    return LogisticRegression(solver=log_reg_solver, max_iter=max_iter)
        
def logistic_regression(df, in_features, out_features, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(df[in_features], df[out_features], train_size=0.75)
    clf_lr = log_model(max_iter=400)
    clf_lr.fit(X_train, y_train.values.ravel())
    if verbose:
        print(f"Accuracy: {clf_lr.score(X_test, y_test)}")
    return clf_lr

def poly_log_model(degree, log_reg_solver='lbfgs', interaction_terms_only=False):
    poly_model = PolynomialFeatures(degree=degree, interaction_only=interaction_terms_only, include_bias=True)
    lr_model = log_model(log_reg_solver=log_reg_solver, max_iter=800)
    return Pipeline([('poly_model', poly_model), ('log_reg_model', lr_model)])

def polynomial_logistic_regression(df, in_features, out_features, degree=2, verbose=False):
    pipeline = poly_log_model(degree)
    
    X_train, X_test, y_train, y_test = train_test_split(df[in_features], df[out_features], train_size=0.75)
    pipeline.fit(X_train, y_train.values.ravel())
    if verbose:
        print(f"Accuracy: {pipeline.score(X_test, y_test)}")
    return pipeline

def xgb_model(max_depth=3, max_leaves=8, learning_rate=0.1, **kwargs):
    return xgb.XGBClassifier(max_depth=max_depth, max_leaves=max_leaves, learning_rate=learning_rate, **kwargs)

def xgboost(df, in_features, out_features, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(df[in_features], df[out_features], train_size=0.75)
    xgb_clf = xgb_model()
    xgb_clf.fit(X_train, y_train.values.ravel())
    if verbose:
        print(f"Accuracy: {xgb_clf.score(X_test, y_test)}")
    return xgb_clf

if __name__ == "__main__":
    df = get_cleaned_shot_data()

    in_features = ["SHOT_CLOCK", "DRIBBLES", "SHOT_DIST", "CLOSE_DEF_DIST", "TOT_TIME_LEFT", "GAME_CLOCK"]
    out_features = ["FGM"]

    ### Uncomment to get best features
    # best_score, best_in_features = select_in_features(LogisticRegression(), df, in_features, out_features, verbose=True)

    best_in_features = ['DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST']

    lr_model = logistic_regression(df, best_in_features, out_features, verbose=True)
    print(lr_model.coef_)

    poly_lr_model = polynomial_logistic_regression(df, best_in_features, out_features, verbose=True)
    print(poly_lr_model[1].coef_)

    xgb_model = xgboost(df, best_in_features, out_features, verbose=True)
    print(xgb_model.feature_importances_)