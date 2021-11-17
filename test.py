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
good = pd.read_csv('good.csv', na_values=['NA'], sep=',', low_memory=False)

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
ignoredCols = ['account_id', 'code', 'disp_id', 'client_id', 'card_id', 'account', 'bank',
                "unemploymant rate '95 ", "no. of commited crimes '95 ", 'issued',
                'name ', 'operation', 'type_y', 'region', 'type', 'trans_id',
                'district_id_x', 'district_id_y']

for key in train_data.keys():
    if key not in ignoredCols:
        col_list.append(key)

train_data = train_data[col_list]
test_data = test_data[col_list]

type_x_vals_train = train_data['type_x'].unique()
type_x_vals_test = test_data['type_x'].unique()

type_x_vals = list(type_x_vals_train)
type_x_vals.extend(x for x in type_x_vals_test if x not in type_x_vals)

train_data = pd.get_dummies(train_data, columns=['type_x'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['type_x'], dtype=bool)

for col in type_x_vals:
    if col not in train_data.columns:
        train_data["type_x_" + col] = False
    elif col not in test_data.columns:
        test_data["type_x_" + col] = False

train_data["k_symbol"].fillna('no info', inplace=True)
test_data["k_symbol"].fillna('no info', inplace=True)

for index, row in train_data.iterrows():
    if train_data.loc[index, 'k_symbol'] == ' ':
        train_data.loc[index, 'k_symbol'] = 'no info'
for index, row in test_data.iterrows():
    if test_data.loc[index, 'k_symbol'] == ' ':
        test_data.loc[index, 'k_symbol'] = 'no info'
        
k_symbol_vals_train = train_data['k_symbol'].unique()
k_symbol_vals_test = test_data['k_symbol'].unique()

k_symbol_vals = list(k_symbol_vals_train)
k_symbol_vals.extend(x for x in k_symbol_vals_test if x not in k_symbol_vals)

train_data = pd.get_dummies(train_data, columns=['k_symbol'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['k_symbol'], dtype=bool)

for col in k_symbol_vals:
    if col not in train_data.columns:
        train_data["k_symbol_" + col] = False
    elif col not in test_data.columns:
        test_data["k_symbol_" + col] = False  

train_data = pd.get_dummies(train_data, columns=['frequency'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['frequency'], dtype=bool)

train_data = train_data.dropna()
competition_inputs = test_data.drop(columns=["loan_id", "status"])
test_data = test_data.drop(columns=["status"])
test_data = test_data.dropna() # test_data must always have 354 rows, keeping it there to make sure
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

y_train_temp = pd.DataFrame(y_train, columns=['status'])
X_train.reset_index(drop=True, inplace=True)
y_train_temp.reset_index(drop=True, inplace=True)
all_train_data = pd.concat([X_train, y_train_temp], axis=1, join='outer')

default_data = all_train_data.loc[all_train_data['status'] == -1]
LIMIT = len(default_data)
non_default_data = all_train_data.loc[all_train_data['status'] == 1]
non_default_data = non_default_data[:LIMIT]

all_train_data = pd.concat([default_data, non_default_data])

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
competition_inputs = scaler.fit_transform(competition_inputs)

dt_classifier = AdaBoostClassifier(random_state=1)

dt_grid_search = GridSearchCV(dt_classifier,
                            param_grid={},
                            scoring='roc_auc',
                            cv=10)

dt_grid_search.fit(X_train, y_train)
best_score = dt_grid_search.best_score_
print("Best Score: " + str(best_score))

predictions_train = dt_grid_search.predict(X_train)
predictions_test = dt_grid_search.predict(X_test)


print('Best score: {}'.format(dt_grid_search.best_score_))
print('Best parameters: {}'.format(dt_grid_search.best_params_))
#cv_scores = cross_val_score(svm_classifier, all_inputs, all_labels, cv=cross_validation, scoring="roc_auc")
#print('Mean ROC AUC: %.3f' % np.mean(cv_scores))

predictions_competition = dt_grid_search.predict_proba(competition_inputs)
predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted1', 'col2'])
predictions_competition.drop('col2', axis=1, inplace=True)
dataframetemp = pd.DataFrame(all_ids_comp, columns=['Id'])
dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
results = dataframeids.drop_duplicates(subset=['Id'], keep='first')

results = results.merge(good, on='Id')

results.to_csv('out.csv', index = False)