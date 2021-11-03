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

### Cleaning Data

# Reading the data
account_data = pd.read_csv('data/account.csv', na_values=['NA'], sep=';', low_memory=False)
client_data = pd.read_csv('data/client.csv', na_values=['NA'], sep=';', low_memory=False)
disp_data = pd.read_csv('data/disp.csv', na_values=['NA'], sep=';', low_memory=False)
district_data = pd.read_csv('data/district.csv', na_values=['NA'], sep=';', low_memory=False)
card_train_data = pd.read_csv('data/card_train.csv', na_values=['NA'], sep=';', low_memory=False)
loan_train_data = pd.read_csv('data/loan_train.csv', na_values=['NA'], sep=';', low_memory=False)
trans_train_data = pd.read_csv('data/trans_train.csv', na_values=['NA'], sep=';', low_memory=False)
card_test_data = pd.read_csv('data/card_test.csv', na_values=['NA'], sep=';', low_memory=False)
loan_test_data = pd.read_csv('data/loan_test.csv', na_values=['NA'], sep=';', low_memory=False)
trans_test_data = pd.read_csv('data/trans_test.csv', na_values=['NA'], sep=';', low_memory=False)

# Removing unnecessary columns
account_data = account_data.drop(columns=["date"])
loan_train_data = loan_train_data.drop(columns=["date"])
loan_test_data = loan_test_data.drop(columns=["date"])

# Merging the data

district_data = district_data.rename(columns={"code ":"code"})

train_data = loan_train_data.merge(trans_train_data, on='account_id', how='inner')
train_data = train_data.merge(account_data, on=['account_id'], how='inner')
train_data = train_data.dropna()
train_data = train_data.merge(district_data, left_on='district_id', right_on='code', how='inner')
train_data = train_data.merge(disp_data, on='account_id', how='inner')
train_data = train_data.merge(card_train_data, on='disp_id', how='outer')
train_data = train_data.merge(client_data, on='client_id', how='inner')

test_data = loan_test_data.merge(trans_test_data, on='account_id', how='inner')
test_data = test_data.merge(account_data, on=['account_id'], how='inner')
test_data = test_data.merge(district_data, left_on='district_id', right_on='code', how='inner')
test_data = test_data.merge(disp_data, on='account_id', how='inner')
test_data = test_data.merge(card_test_data, on='disp_id', how='outer')
test_data = test_data.merge(client_data, on='client_id', how='inner')

# Removing NaN values

col_list = []
ignoredCols = ['name ', 'region', 'card_id', 'type_x', 'issued', 'district_id_x', 'client_id',
                'district_id_y', 'disp_id', 'account_id', 'trans_id', 'operation', 'bank', 'type',
                'k_symbol', 'account']

for key in train_data.keys():
    if key not in ignoredCols:
        col_list.append(key)

train_data = train_data[col_list]
test_data = test_data[col_list]
# test_data = test_data.drop(columns=["status"])

train_data["unemploymant rate '95 "] = pd.to_numeric(train_data["unemploymant rate '95 "], errors='coerce')
train_data["no. of commited crimes '95 "] = pd.to_numeric(train_data["no. of commited crimes '95 "], errors='coerce')
train_data["unemploymant rate '95 "] = pd.to_numeric(train_data["unemploymant rate '95 "], errors='coerce')
train_data["no. of commited crimes '95 "] = pd.to_numeric(train_data["no. of commited crimes '95 "], errors='coerce')

test_data["unemploymant rate '95 "] = pd.to_numeric(test_data["unemploymant rate '95 "], errors='coerce')
test_data["no. of commited crimes '95 "] = pd.to_numeric(test_data["no. of commited crimes '95 "], errors='coerce')
test_data["unemploymant rate '95 "] = pd.to_numeric(test_data["unemploymant rate '95 "], errors='coerce')
test_data["no. of commited crimes '95 "] = pd.to_numeric(test_data["no. of commited crimes '95 "], errors='coerce')

grade_dic = {'monthly issuance':1, 'weekly issuance':2, 'issuance after transaction':3}
train_data["frequency"].replace(grade_dic, inplace=True)
test_data["frequency"].replace(grade_dic, inplace=True)

grade_dic = {'OWNER':1, 'DISPONENT':2}
train_data["type_y"].replace(grade_dic, inplace=True)
test_data["type_y"].replace(grade_dic, inplace=True)

# test_data = test_data.drop_duplicates(subset=['loan_id'], keep='first')

train_data["unemploymant rate '95 "].fillna((train_data["unemploymant rate '95 "].mean()), inplace=True)
train_data["no. of commited crimes '95 "].fillna((train_data["no. of commited crimes '95 "].mean()), inplace=True)

test_data["unemploymant rate '95 "].fillna((test_data["unemploymant rate '95 "].mean()), inplace=True)
test_data["no. of commited crimes '95 "].fillna((test_data["no. of commited crimes '95 "].mean()), inplace=True)

train_data = train_data.dropna()
competition_inputs = test_data.drop(columns=["loan_id", "status"])
test_data = test_data.drop(columns=["status"])
# test_data = test_data.dropna() # test_data must always have 354 rows, keeping it there to make sure
all_ids_comp = test_data['loan_id'].values

default_data = train_data.loc[train_data['status'] == -1]
LIMIT = len(default_data)
non_default_data = train_data.loc[train_data['status'] == 1]
non_default_data = non_default_data[:LIMIT]

train_data = pd.concat([default_data, non_default_data])

all_inputs = train_data.drop(columns=["status", "loan_id"])
all_labels = train_data['status'].values
all_ids = train_data['loan_id'].values

competition_inputs = test_data.drop(columns=["loan_id"])
all_ids_comp = test_data['loan_id'].values

X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_labels, test_size=0.25)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
competition_inputs = scaler.fit_transform(competition_inputs)

cross_validation = StratifiedKFold(n_splits=10)

svm_classifier = SVC(random_state=1, probability=True)

parameters = [{'kernel': ['rbf', 'linear'], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100]}]

grid_search = GridSearchCV(svm_classifier,
                    parameters,
                    n_jobs=4,
                    scoring='accuracy',
                    cv=cross_validation)

grid_search.fit(X_train, y_train)


predictions_train = grid_search.predict(X_train)
predictions_test = grid_search.predict(X_test)

predictions_competition = grid_search.predict_proba(competition_inputs)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_comp, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')

results.to_csv('out.csv', index = False)