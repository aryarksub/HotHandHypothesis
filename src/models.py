from itertools import combinations
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression

from data_processor import get_cleaned_shot_data

def select_in_features(df, all_in_features, out_features, max_in_features=None, verbose=False):
    best_score = (0, [])
    lr_model = LogisticRegression()
    kfold_cv = KFold(n_splits=10)

    out_values = df[out_features].values.ravel()

    if max_in_features is None:
        max_in_features = len(all_in_features)

    for num_features in range(1, max_in_features+1):
        all_in_feature_combinations = combinations(all_in_features, num_features)
        for in_features in all_in_feature_combinations:
            in_features = list(in_features)
            scores = cross_val_score(lr_model, df[in_features].values, out_values, cv=kfold_cv)
            accuracy = np.mean(scores)
            if accuracy > best_score[0]:
                best_score = (accuracy, in_features)
        if verbose:
            print(f"Done with all combinations of {num_features} input features")
            print(f"Current best: {best_score}")
    return best_score
        
def logistic_regression(df, in_features, out_features, verbose=False):
    X_train, X_test, y_train, y_test = train_test_split(df[in_features], df[out_features], train_size=0.75)
    clf_lr = LogisticRegression()
    clf_lr.fit(X_train, y_train.values.ravel())
    if verbose:
        print(f"Accuracy: {clf_lr.score(X_test, y_test)}")
    return clf_lr

df = get_cleaned_shot_data()

in_features = ["SHOT_CLOCK", "DRIBBLES", "SHOT_DIST", "CLOSE_DEF_DIST", "TOT_TIME_LEFT", "GAME_CLOCK"]
out_features = ["FGM"]

### Uncomment to get best features
# best_score, best_in_features = select_in_features(df, in_features, out_features, verbose=True)

best_in_features = ['DRIBBLES', 'SHOT_DIST', 'CLOSE_DEF_DIST', 'TOT_TIME_LEFT', 'GAME_CLOCK']

lr_model = logistic_regression(df, best_in_features, out_features, verbose=True)
print(lr_model.coef_)


