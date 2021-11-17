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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
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

# Cleaning the data

account_data["creation_year"] = 0 
account_data["creation_month"] = 0 
account_data["creation_day"] = 0
for i in range(0, len(account_data)):
    year = account_data.iloc[i, 3] // 10000
    if year < 21:
        year = year + 2000
    else:
        year = year + 1900
    account_data.iloc[i, 4] = year
    account_data.iloc[i, 5] = (account_data.iloc[i, 3] % 10000) // 100
    account_data.iloc[i, 6] = account_data.iloc[i, 3] % 100
    
account_data = account_data.drop(columns=['date'])

loan_train_data["loan_year"] = 0 
loan_train_data["loan_month"] = 0 
loan_train_data["loan_day"] = 0
for i in range(0, len(loan_train_data)):
    year = loan_train_data.iloc[i, 2] // 10000
    if year < 21:
        year = year + 2000
    else:
        year = year + 1900
    loan_train_data.iloc[i, 7] = year
    loan_train_data.iloc[i, 8] = (loan_train_data.iloc[i, 2] % 10000) // 100
    loan_train_data.iloc[i, 9] = loan_train_data.iloc[i, 2] % 100
    
loan_train_data = loan_train_data.drop(columns=['date'])
train_data = loan_train_data.merge(account_data, on='account_id', how='inner')

train_data['days_since_creation'] = 0

for i in range(0, len(train_data)):
    loan_date = train_data.iloc[i,6]*365 + train_data.iloc[i, 7]*31 + train_data.iloc[i,8]
    creation_date = train_data.iloc[i,-4]*365 + train_data.iloc[i, -3]*31 + train_data.iloc[i,-2]
    train_data.iloc[i, -1] = loan_date - creation_date 

client_data["gender"] = ""
client_data["birth_year"] = 0 
client_data["birth_month"] = 0 
client_data["birth_day"] = 0
for i in range(0, len(client_data)):
    year = client_data.iloc[i, 1] // 10000
    if year < 21:
        year = year + 2000
    else:
        year = year + 1900
    client_data.iloc[i, 4] = year
    if ((client_data.iloc[i, 1] % 10000) // 100) > 50:
        client_data.iloc[i, 3] = "F"  
        client_data.iloc[i, 5] = (client_data.iloc[i, 1] % 10000) // 100 - 50
    else:
        client_data.iloc[i, 3] = "M"  
        client_data.iloc[i, 5] = (client_data.iloc[i, 1] % 10000) // 100
    client_data.iloc[i, 6] = client_data.iloc[i, 1] % 100

client_data = client_data.merge(disp_data, on='client_id', how='inner')
train_data = train_data.drop(columns=["district_id"])
train_data = train_data.merge(client_data, on=['account_id'], how='inner')
train_data = train_data.merge(district_data, left_on='district_id', right_on="code ", how='inner')
# train_data = train_data.merge(trans_train_data, on='account_id', how="left")

loan_test_data["loan_year"] = 0 
loan_test_data["loan_month"] = 0 
loan_test_data["loan_day"] = 0
for i in range(0, len(loan_test_data)):
    year = loan_test_data.iloc[i, 2] // 10000
    if year < 21:
        year = year + 2000
    else:
        year = year + 1900
    loan_test_data.iloc[i, 7] = year
    loan_test_data.iloc[i, 8] = (loan_test_data.iloc[i, 2] % 10000) // 100
    loan_test_data.iloc[i, 9] = loan_test_data.iloc[i, 2] % 100
    
loan_test_data = loan_test_data.drop(columns=['date'])
test_data = loan_test_data.merge(account_data, on='account_id', how='inner')

test_data['days_since_creation'] = 0

for i in range(0, len(test_data)):
    loan_date = test_data.iloc[i,6]*365 + test_data.iloc[i, 7]*31 + test_data.iloc[i,8]
    creation_date = test_data.iloc[i,-4]*365 + test_data.iloc[i, -3]*31 + test_data.iloc[i,-2]
    test_data.iloc[i, -1] = loan_date - creation_date 

test_data = test_data.drop(columns=["district_id"])
test_data = test_data.merge(client_data, on=['account_id'], how='inner')
test_data = test_data.merge(district_data, left_on='district_id', right_on="code ", how='inner')
# test_data = test_data.rename(columns={"amount":"loaned_amount", "type":"account_type"})
# test_data = test_data.merge(trans_test_data, on='account_id', how="left")

train_data = pd.get_dummies(train_data, columns=['frequency'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['frequency'], dtype=bool)

train_data = pd.get_dummies(train_data, columns=['gender'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['gender'], dtype=bool)

# train_data = pd.get_dummies(train_data, columns=['account_type'], dtype=bool)
# test_data = pd.get_dummies(test_data, columns=['account_type'], dtype=bool)

# train_data = pd.get_dummies(train_data, columns=['type'], dtype=bool)
# test_data = pd.get_dummies(test_data, columns=['type'], dtype=bool)

# train_data = pd.get_dummies(train_data, columns=['operation'], dtype=bool)
# test_data = pd.get_dummies(test_data, columns=['operation'], dtype=bool)

train_data = train_data.drop(columns=["name ", "region", "type"])
test_data = test_data.drop(columns=["name ", "region", "type"])

train_data["unemploymant rate '95 "] = pd.to_numeric(train_data["unemploymant rate '95 "], downcast="float", errors='coerce')
test_data["unemploymant rate '95 "] = pd.to_numeric(test_data["unemploymant rate '95 "], downcast="float", errors='coerce')

train_data["no. of commited crimes '95 "] = pd.to_numeric(train_data["no. of commited crimes '95 "], downcast="float", errors='coerce')
test_data["no. of commited crimes '95 "] = pd.to_numeric(test_data["no. of commited crimes '95 "], downcast="float", errors='coerce')

train_data["unemploymant rate '95 "].fillna((train_data["unemploymant rate '95 "].mean()), inplace=True)
train_data["no. of commited crimes '95 "].fillna((train_data["no. of commited crimes '95 "].mean()), inplace=True)


# train_data = train_data.drop(columns=["account_id", "client_id", "district_id", "disp_id",
#                                       "trans_id", "date", "k_symbol", "account", "bank"])

# amount_means = list(train_data.groupby(['loan_id'])['amount'].mean())
# balance_means = list(train_data.groupby(['loan_id'])['balance'].mean())

train_data = train_data.drop_duplicates(subset=['loan_id'], keep='first')
test_data = test_data.drop_duplicates(subset=['loan_id'], keep='first')

# cnt = 0
# for index, row in train_data.iterrows():
#     train_data.loc[index, 'amount'] = amount_means[cnt]
#     train_data.loc[index, 'balance'] = balance_means[cnt]
#     cnt = cnt + 1
    
# test_data = test_data.drop(columns=["account_id", "client_id", "district_id", "disp_id",
#                                       "trans_id", "date", "k_symbol", "account", "bank"])

# amount_means = list(test_data.groupby(['loan_id'])['amount'].mean())
# balance_means = list(test_data.groupby(['loan_id'])['balance'].mean())

# test_data = test_data.drop_duplicates(subset=['loan_id'], keep='first')

# cnt = 0
# for index, row in test_data.iterrows():
#     test_data.loc[index, 'amount'] = amount_means[cnt]
#     test_data.loc[index, 'balance'] = balance_means[cnt]
#     cnt = cnt + 1




# train_data.to_csv('train.csv', index = False)
# test_data.to_csv('test.csv', index = False)

competition_inputs = test_data.drop(columns=["loan_id", "status"])
test_data = test_data.drop(columns=["status"])
test_data = test_data.dropna() # test_data must always have 354 rows, keeping it there to make sure
all_ids_comp = test_data['loan_id'].values

# default_data = train_data.loc[train_data['status'] == -1]
# LIMIT = len(default_data)
# non_default_data = train_data.loc[train_data['status'] == 1]
# non_default_data = non_default_data[:LIMIT]

# train_data = pd.concat([default_data, non_default_data])

all_inputs = train_data.drop(columns=["status", "loan_id"])
all_labels = train_data['status'].values
all_ids = train_data['loan_id'].values

competition_inputs = test_data.drop(columns=["loan_id"])
all_ids_comp = test_data['loan_id'].values

X_train, X_test, y_train, y_test = train_test_split(all_inputs, all_labels, test_size=0.25)

# y_train_temp = pd.DataFrame(y_train, columns=['status'])
# X_train.reset_index(drop=True, inplace=True)
# y_train_temp.reset_index(drop=True, inplace=True)
# all_train_data = pd.concat([X_train, y_train_temp], axis=1, join='outer')

# default_data = all_train_data.loc[all_train_data['status'] == -1]
# LIMIT = len(default_data)
# non_default_data = all_train_data.loc[all_train_data['status'] == 1]
# non_default_data = non_default_data[:LIMIT]

# all_train_data = pd.concat([default_data, non_default_data])

# classifier = AdaBoostClassifier()

# grid_search = GridSearchCV(classifier, scoring="roc_auc", cv=10, param_grid={})
# grid_search.fit(X_train, y_train)

# predictions_train = grid_search.predict(X_train)
# predictions_test = grid_search.predict(X_test)

# predictions_competition = grid_search.predict_proba(competition_inputs)
# predictions_competition = pd.DataFrame(predictions_competition, columns=['Predicted', 'col2'])
# predictions_competition.drop('col2', axis=1, inplace=True)
# dataframetemp = pd.DataFrame(all_ids_comp, columns=['Id'])
# dataframeids = pd.concat([dataframetemp, predictions_competition], axis=1)
# results = dataframeids.drop_duplicates(subset=['Id'], keep='first')

# print("auc: {}".format(grid_search.best_score_))

# results = results.merge(good, on='Id')

# results.to_csv('out.csv', index = False)
