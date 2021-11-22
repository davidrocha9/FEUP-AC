import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import sklearn.tree as tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn import tree, svm, preprocessing
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Reading the data
train = pd.read_csv('train.csv', na_values=['NA'], low_memory=False)
test = pd.read_csv('test.csv', na_values=['NA'], low_memory=False)
good = pd.read_csv('good.csv', na_values=['NA'], low_memory=False)
mygood = pd.read_csv('mygood.csv', na_values=['NA'], low_memory=False)

inputs = train.drop(columns=['status', 'loan_id'])
labels = train['status']

#Split in train and test - Should also try split manually by date
x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3, random_state=1)

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train, y_train)

# Use for KNN
# scaler = preprocessing.StandardScaler().fit(x_train)
# x_train = scaler.transform(x_train)

#Train a model
# classifier = KNeighborsClassifier()
# classifier = tree.DecisionTreeClassifier()
# classifier = svm.SVC(probability=True)
# classifier = RandomForestClassifier(300)
# classifier = xgb.XGBClassifier()
# classifier = MLPClassifier(alpha=1, max_iter=1000)
classifier = AdaBoostClassifier()
# classifier = GaussianNB()
# classifier = VotingClassifier(
#     estimators=[('dt', tree.DecisionTreeClassifier()), ('svm', svm.LinearSVC()), ('xgb', xgb.XGBClassifier())],
#     voting='hard', weights=[1,1,1]
# )
# classifier = LogisticRegression()


# Feature Selection
# classifier = RFECV(classifier, scoring='roc_auc')

classifier.fit(x_train, y_train)
dt_grid_search = GridSearchCV(classifier, scoring="roc_auc", cv=5, param_grid={})
dt_grid_search.fit(x_train, y_train)

# print("Selected Features: %s" % (classifier.ranking_))




# Test and display stats

# Use for KNN
# x_test = scaler.transform(x_test)

# Save Decision Tree
# fig = plt.figure(figsize=(50,50))
# tree.plot_tree(classifier, feature_names=train_data.columns[2:], class_names=['-1', '1'], label='root', filled=True, proportion=True)
# fig.savefig("decistion_tree.png")

# plot_confusion_matrix(classifier, x_test, y_test)
# plt.show()

# Print Metrics    
print("Area under ROC curve: " + str(roc_auc_score(y_test, classifier.predict(x_test))))
print("Accuracy: " + str(accuracy_score(y_test, classifier.predict(x_test))))
print("Precision: " + str(precision_score(y_test, classifier.predict(x_test))))
print("Recall: " + str(recall_score(y_test, classifier.predict(x_test))))
print("f1: " + str(f1_score(y_test, classifier.predict(x_test))))

test_inputs = test.drop(columns=['status', 'loan_id'])
test_ids = test['loan_id']

predictions_competition = dt_grid_search.predict_proba(test_inputs)
predictions_competition = pd.DataFrame(predictions_competition, columns=['col2', 'Predicted'])
predictions_competition.drop('col2', axis=1, inplace=True)
test_ids = test_ids.to_frame()
results = pd.concat([test_ids, predictions_competition], axis=1)
results = results.rename(columns={"loan_id":"Id"})
# mygood = mygood.rename(columns={"Predicted":"Predicted2"})
# results = results.merge(mygood, on="Id", how="inner")
# results = results.merge(good, on="Id", how="inner")

results.to_csv('out.csv', index = False)