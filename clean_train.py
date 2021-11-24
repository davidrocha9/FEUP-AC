import pandas as pd

### Cleaning Data

# Reading the data
account_data = pd.read_csv('data/account.csv', na_values=['NA'], sep=';', low_memory=False)
client_data = pd.read_csv('data/client.csv', na_values=['NA'], sep=';', low_memory=False)
disp_data = pd.read_csv('data/disp.csv', na_values=['NA'], sep=';', low_memory=False)
district_data = pd.read_csv('data/district.csv', na_values=['NA'], sep=';', low_memory=False)
card_train_data = pd.read_csv('data/card_train.csv', na_values=['NA'], sep=';', low_memory=False)
loan_train_data = pd.read_csv('data/loan_train.csv', na_values=['NA'], sep=';', low_memory=False)
trans_train_data = pd.read_csv('data/trans_train.csv', na_values=['NA'], sep=';', low_memory=False)
    

# out = pd.read_csv('train.csv', na_values=['NA'], sep=',', low_memory=False)

from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

# Cleaning the data

loan_train_data = loan_train_data.rename(columns={"amount":"loan_amount"})
client_data = client_data.rename(columns={"type":"client_type"})
trans_train_data = trans_train_data.rename(columns={"type":"trans_type", "amount":"trans_amount", "date":"trans_date"})

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
    
account_ids = list(train_data['account_id'].unique())
trans_train_data = trans_train_data[trans_train_data['account_id'].isin(account_ids)]

for index, row in trans_train_data.iterrows():
    rows = trans_train_data[trans_train_data['account_id'] == trans_train_data.loc[index, 'account_id']]
    trans_train_data.loc[index, 'withdrawals'] = len(rows[rows['trans_type'] == 'withdrawal']) + len(rows[rows['trans_type'] == 'withdrawal in cash'])
    trans_train_data.loc[index, 'credits'] = len(rows[rows['trans_type'] == 'credit'])
    
    trans_train_data.loc[index, 'nr_movements'] = len(rows)
    sum_amount = sum(rows['trans_amount'])
    max_date = max(rows['trans_date'])
    min_date = min(rows['trans_date'])
    
    if trans_train_data.loc[index, 'trans_date'] == max_date:
        trans_train_data.loc[index, 'balance_last'] = trans_train_data.loc[index, 'balance']
        trans_train_data.loc[index, 'amount_last'] = trans_train_data.loc[index, 'trans_amount']
    
    max_year = max_date // 10000
    if max_year < 21:
        max_year = max_year + 2000
    else:
        max_year = max_year + 1900
    max_month = (max_date % 10000) // 100
    max_day = max_date % 100
    
    min_year = min_date // 10000
    if min_year < 21:
        min_year = min_year + 2000
    else:
        min_year = min_year + 1900
    min_month = (min_date % 10000) // 100
    min_day = min_date % 100
    
    trans_train_data.loc[index, 'trans_date_diff'] = days_between(str(max_year) + "-" + str(max_month) + "-" + str(max_day), str(min_year) + "-" + str(min_month) + "-" + str(min_day))
    trans_train_data.loc[index, 'amount_month'] = sum_amount / (trans_train_data.loc[index, 'trans_date_diff'] / 30.0)
    
    trans_train_data.loc[index, 'min_trans_amount'] = min(rows['trans_amount'])
    trans_train_data.loc[index, 'max_trans_amount'] = max(rows['trans_amount'])
    trans_train_data.loc[index, 'avg_trans_amount'] = rows['trans_amount'].mean()
    trans_train_data.loc[index, 'range_amount'] = trans_train_data.loc[index, 'max_trans_amount'] - trans_train_data.loc[index, 'min_trans_amount']
    
    trans_train_data.loc[index, 'min_trans_balance'] = min(rows['balance'])
    trans_train_data.loc[index, 'max_trans_balance'] = max(rows['balance'])
    trans_train_data.loc[index, 'avg_trans_balance'] = rows['balance'].mean()
    trans_train_data.loc[index, 'range_balance'] = trans_train_data.loc[index, 'max_trans_balance'] - trans_train_data.loc[index, 'min_trans_balance']    
    
trans_train_data = trans_train_data.drop(columns=['trans_type', 'operation', "k_symbol", "account", "bank"])
trans_train_data = trans_train_data.dropna()
trans_train_data = trans_train_data.drop_duplicates(subset=['account_id'], keep='first')
train_data = train_data.merge(trans_train_data, on='account_id', how="left")
train_data = train_data[train_data.std(axis=1) > 0]

train_data['members'] = 0
train_data['owner_age'] = 0
train_data['able_to_pay'] = False

for index, row in train_data.iterrows():
    trans_rows = train_data[train_data['loan_id'] == train_data.loc[index,'loan_id']]
    train_data.loc[index, 'members'] = len(trans_rows['type'].unique())
    
    if train_data.loc[index, 'amount_month'] > train_data.loc[index, 'payments']:
        train_data.loc[index, 'able_to_pay'] = True
        
    
train_data = train_data.sort_values('loan_id')

train_data = train_data.drop(columns=["account_id", "client_id", "birth_number", "district_id", "disp_id",
                                      "trans_id", "trans_date"])

train_data = train_data[train_data['type'] == 'OWNER']

train_data = train_data.drop_duplicates(subset=['loan_id'], keep='first')

train_data = train_data.drop(columns=['name ', 'region', 'type'])

train_data = pd.get_dummies(train_data, columns=['frequency'], dtype=bool)
train_data = pd.get_dummies(train_data, columns=['gender'], dtype=bool)
    
tf = train_data[["unemploymant rate '95 ", "no. of commited crimes '95 "]]
tf["unemploymant rate '95 "] = pd.to_numeric(tf["unemploymant rate '95 "], downcast="float", errors='coerce')
tf["no. of commited crimes '95 "] = pd.to_numeric(tf["no. of commited crimes '95 "], downcast="float", errors='coerce')
tf = tf.interpolate()

train_data["unemploymant rate '95 "] = tf["unemploymant rate '95 "]
train_data["no. of commited crimes '95 "] = tf["no. of commited crimes '95 "]

train_data = train_data.drop(columns=['loan_day'])

for index, row in train_data.iterrows():
    train_data.loc[index, 'owner_age'] = train_data.loc[index, 'loan_year'] - train_data.loc[index, 'birth_year']
    
train_data = train_data.drop(columns=["birth_year", "birth_month", "birth_day", "code ",
                        "creation_year", "creation_month", "creation_day",
                        "trans_amount", "balance"])

train_data.to_csv('train.csv', index = False)
