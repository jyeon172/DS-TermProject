import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = None
pd.options.display.width = None


def do_analysis(df, indicators, target_columns):
    # Set features and targets
    features = df[indicators]
    targets = df[target_columns]

    features['Total_Symptom'] = features[indicators].apply(get_symptom_count, axis=1)

    # 1: Logistic Regression, 2: Decision Tree Classifier, 3: Random Forest Classifier
    algo = [1, 2, 3]

    # Analysis for each Severity with 3 kinds of algorithm
    # Severity - Severity_None, Severity_Mild, Severity_Moderate, Severity_Severe
    # Analysis - Logistic Regression, Decision Tree Classifier, Random Forest Classifier
    for i in range(len(target_columns)):
        for j in range(len(algo)):
            print("====== For {} ======".format(target_columns[i]))
            target(features, targets, target_columns[i], algo[j])
            print()


def target(fs, ts, t, algo):
    d = fs
    d[t] = ts[t].values  # set current target feature

    # print("== print data ====")
    # print(d)
    # print()

    x = d.drop([t, 'Total_Symptom'], axis=1)
    x = PCA(n_components=3).fit_transform(x)
    y = d[t]

    # print("== print x ====")
    # print(x)
    # print("== print y ====")
    # print(y)

    do_split(x, y, algo)


def do_split(x, y, algo):
    # divide into test and train dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.3, shuffle=True, stratify=y)

    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)  # K-Fold (for k=10)

    if algo == 1:  # Logistic Regression
        do_LR(x_train, x_test, y_train, y_test, k_fold)
    elif algo == 2:  # Decision Tree Classifier
        do_DTC(x_train, x_test, y_train, y_test, k_fold)
    elif algo == 3:  # Random Forest Classifier
        do_RFC(x_train, x_test, y_train, y_test, k_fold)


# Logistic Regression
# Create a base model: LogisticRegression()
# Create the parameter grid based on the results of random search: GridSearchCV(,,,)
# params grid = {
#     "penalty": ['l1', 'l2', 'elasticnet', 'none'],
#     "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
# }
# Create a tuned model with grid search: .best_estimator_
# code resource : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def do_LR(x_train, x_test, y_train, y_test, k_fold):
    print("<< Logistic Regression >>")
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    print("> Logistic Regression score: {}".format(lr.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Logistic Regression
    print("> Confusion Matrix")
    do_cm(lr, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }
    lr_reg = GridSearchCV(lr, params, cv=10, n_jobs=10)
    lr_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(lr_reg.best_estimator_))

    # Set model with best estimator
    lr_tune = lr_reg.best_estimator_
    lr_tune.fit(x_train, y_train)
    print("> Logistic Regression tuned score: {}".format(lr_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Logistic Regression
    print("> Confusion Matrix using tuned")
    do_cm(lr_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(lr_tune, x_test, y_test, k_fold)


# code resource : https://www.kaggle.com/code/sararasoulian/classification
def do_DTC(x_train, x_test, y_train, y_test, k_fold):  # Decision Tree Classifier
    print("<< Decision Tree Classifier >>")
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    print("> Decision Tree Classifier score: {}".format(dtc.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Decision Tree Classifier
    print("> Confusion Matrix")
    do_cm(dtc, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "criterion": ["gini", "entropy"],
        "max_depth": [15, 20, 25],
    }
    dtc_reg = GridSearchCV(dtc, params, cv=10, n_jobs=10)
    dtc_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(dtc_reg.best_estimator_))

    # Set model with best estimator
    dtc_tune = dtc_reg.best_estimator_
    dtc_tune.fit(x_train, y_train)
    print("> Decision Tree Classifier tuned score: {}".format(dtc_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Decision Tree Classifier
    print("> Confusion Matrix using tuned")
    do_cm(dtc_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(dtc_tune, x_test, y_test, k_fold)


# code resource : https://www.kaggle.com/code/sararasoulian/classification
def do_RFC(x_train, x_test, y_train, y_test, k_fold):  # Random Forest Classifier
    print("<< Random Forest Classifier >>")
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    print("> Random Forest Classifier score: {}".format(rfc.score(x_test, y_test)))  # Print result

    # Print confusion matrix for Random Forest Classifier model
    print("> Confusion Matrix")
    do_cm(rfc, x_test, y_test)

    # Find the best estimator using Grid Search
    params = {
        "max_depth": [15, 20, 25],
        "n_estimators": [27, 30, 33],
        "criterion": ["gini", "entropy"],
    }
    rf_reg = GridSearchCV(rfc, params, cv=10, n_jobs=10)
    rf_reg.fit(x_train, y_train)
    print("> The best estimator using GridSearch: {}".format(rf_reg.best_estimator_))

    # Set model with best estimator
    rfc_tune = rf_reg.best_estimator_
    rfc_tune.fit(x_train, y_train)
    print("> Random Forest Classifier tuned score: {}".format(rfc_tune.score(x_test, y_test)))  # Print result

    # Print confusion matrix for tuned Random Forest Classifier model
    print("> Confusion Matrix using tuned")
    do_cm(rfc_tune, x_test, y_test)

    # Do evaluation with k-fold (k=10)
    do_evaluation(rfc_tune, x_test, y_test, k_fold)


def do_cm(model, x_test, y_test):  # Print Confusion matrix
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)  # Set confusion matrix
    # sns.heatmap(cm, annot=True)
    # plt.show()
    print(cm)  # Print result


def do_evaluation(tune, x_test, y_test, kf):  # Print K-Fold validation Accuracy
    # Do evaluation with K-Fold (k=10)
    scores = cross_val_score(tune, x_test, y_test, cv=kf, n_jobs=1, scoring="accuracy")
    # Print result
    i = 0
    print("< KFold Accuracy >")
    for score in scores:
        i = i + 1
        print("> When {}, Accuracy: {}".format(i, score))
    print("> Cross validation mean score: {}".format(scores.mean()))  # Print mean score
