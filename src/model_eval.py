import numpy as np
from itertools import combinations, product
from sklearn.model_selection import KFold, cross_val_score

from data_processor import get_cleaned_shot_data
from models import log_model, poly_log_model, xgb_model

def evaluate_model(model, df, in_features, out_features, verbose=False):
    kfold_cv = KFold(n_splits=10)
    out_values = df[out_features].values.ravel()
    scores = cross_val_score(model, df[in_features].values, out_values, cv=kfold_cv)
    accuracy = np.mean(scores)
    if verbose:
        print(f"Accuracy: {accuracy}")
    return accuracy

def best_features_for_model(model, df, model_name=None, verbose=False):
    if verbose and model_name is not None:
        print(f"Determining best features for model: {model_name}")

    # keywords for feature columns that are most important for shot quality
    keywords = ["DRIBBLES", "SHOT_DIST", "CLOSE_DEF_DIST", "HEIGHT"]
    all_columns = set(df.columns)
    # separate feature columns based on keywords
    columns = {keyword : [col for col in all_columns if keyword in col] for keyword in keywords}

    out_features = ["FGM"]

    best_score, best_features = 0, []

    # choose one feature column from each keyword
    for feature_list in product(*list(columns.values())):
        score = evaluate_model(model, df, list(feature_list), out_features, verbose=False)
        if score > best_score:
            best_score = score
            best_features = feature_list
            if verbose:
                print(f"New best accuracy ({round(best_score, 5)}) with features: {best_features}")
    return best_score, best_features

if __name__ == '__main__':
    df = get_cleaned_shot_data()

    # Uncomment for best features for simple logistic regression
    # log_reg_accuracy, log_reg_features = best_features_for_model(
    #     log_model(max_iter=400), df, model_name="Logistic regression", verbose=True
    # )
    # print(f"Best logistic regression model:\n  Features: {log_reg_features}\n  Accuracy: {log_reg_accuracy}")

    # Uncomment for best features for polynomial logistic regression (degree = 2)
    # poly_log_reg_accuracy, poly_log_reg_features = best_features_for_model(
    #     poly_log_model(2, interaction_terms_only=True), df, model_name="Poly logistic regression", verbose=True
    # )
    # print(f"Best polynomial logistic regression model:\n  Features: {poly_log_reg_features}\n  Accuracy: {poly_log_reg_accuracy}")

    # Uncomment for best features for XGBoost
    xgb_accuracy, xgb_features = best_features_for_model(
        xgb_model(), df, model_name="XGBoost", verbose=True        
    )
    print(f"Best XGBoost model:\n  Features: {xgb_features}\n  Accuracy: {xgb_accuracy}")