import pandas as panda
import matplotlib.pyplot as plot
import numpy as numpy
import sklearn.cross_validation as sk
import sklearn.preprocessing as imp
import sklearn.naive_bayes as naive
import sklearn as skl
import sklearn.ensemble as ensemble
import sklearn.linear_model as linear

# check if unwanted correlation among features
def plot_correlation(data_table, size=11):
    corr = data_table.corr()
    plot.subplots(figsize=(size, size))
    plot.xticks(range(len(corr.columns)), corr.columns)
    plot.yticks(range(len(corr.columns)), corr.columns)

def analyse_data(data_csv):
    # look at shape
    data_csv.shape
    # look at head
    data_csv.head(5)
    # look at tail
    data_csv.tail(5)
    # look if any value is null
    data_csv.isnull().values.any()
    # look for correlation
    plot_correlation
    # true false ration
    true_false_ratio(data_csv)


def true_false_ratio(data_csv):
    num_true = len(data_csv.loc[data_csv['diabetes'] == 1])
    num_false = len(data_csv.loc[data_csv['diabetes'] == 0])

    print(num_true)
    print(num_false)

def true_false_ratio_numpy(dt):
    num_true = len(dt[dt[:] == 1])
    num_false = len(dt[dt[:] == 0])

    print(num_true)
    print(num_false)

def tf_ratio_test_train(xte,xtr, yte, ytr):
    true_false_ratio_numpy(xte)
    true_false_ratio_numpy(xtr)
    true_false_ratio_numpy(ytr)
    true_false_ratio_numpy(ytr)

def metrics(y, predict):
    result = skl.metrics.accuracy_score(y, predict)
    print("=====")
    print(result)
    print(skl.metrics.classification_report(y, predict))
    print("=====")

data_csv = panda.read_csv("./data/pima-data.csv");
del data_csv['skin']
diabetes_map = {True : 1, False : 0}
data_csv['diabetes'] = data_csv['diabetes'].map(diabetes_map)

# split into test and training
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicated_class_names = ['diabetes']
x = data_csv[feature_col_names].values
y = data_csv[predicated_class_names].values
split_test_size = 0.30
x_train, x_test, y_train, y_test = sk.train_test_split(x, y, test_size=split_test_size, random_state=42)

# check diabets true false ration in train and test
tf_ratio_test_train(x_train, x_test, y_train, y_test)


# Hidden Missing Values
fill_0 = imp.Imputer(missing_values=0, strategy="mean", axis=0)
x_train = fill_0.fit_transform(x_train)
x_test = fill_0.fit_transform(x_test)

## train naive bayes
nb_model = naive.GaussianNB()
nb_model.fit(x_train, y_train.ravel())
nb_predit_train = nb_model.predict(x_train)
nb_predict_test = nb_model.predict(x_test)
# scores
metrics(y_train, nb_predit_train)
metrics(y_test, nb_predict_test)


# ## train random forest
nb_model = ensemble.RandomForestClassifier(random_state=42)
nb_model.fit(x_train, y_train.ravel())
nb_predit_train = nb_model.predict(x_train)
nb_predict_test = nb_model.predict(x_test)
## scores
metrics(y_train, nb_predit_train)
metrics(y_test, nb_predict_test)

## train Logistic regression
nb_model = linear.LogisticRegression(C=0.7, random_state=42)
nb_model.fit(x_train, y_train.ravel())
nb_predit_train = nb_model.predict(x_train)
nb_predict_test = nb_model.predict(x_test)
## scores
metrics(y_train, nb_predit_train)
metrics(y_test, nb_predict_test)

## choose regularization parameter in a while loop, to get the hightest recall score on test.

## run logistic regression with balancing classes inbuilt
## train Logistic regression
nb_model = linear.LogisticRegression(C=0.7, class_weight="balanced", random_state=42)
nb_model.fit(x_train, y_train.ravel())
nb_predit_train = nb_model.predict(x_train)
nb_predict_test = nb_model.predict(x_test)
## scores
metrics(y_train, nb_predit_train)
metrics(y_test, nb_predict_test)

## to get an optimal regularization parameter for both test and train, we need cross validation.
nb_model = linear.LogisticRegressionCV(n_jobs=-1, cv = 10, Cs=3, refit=False, class_weight="balanced", random_state=42)
nb_model.fit(x_train, y_train.ravel())
nb_predit_train = nb_model.predict(x_train)
nb_predict_test = nb_model.predict(x_test)
## scores
metrics(y_train, nb_predit_train)
metrics(y_test, nb_predict_test)