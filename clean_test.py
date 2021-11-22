import pandas as pd

### Cleaning Data

# Reading the data
account_data = pd.read_csv('data/account.csv', na_values=['NA'], sep=';', low_memory=False)
client_data = pd.read_csv('data/client.csv', na_values=['NA'], sep=';', low_memory=False)
disp_data = pd.read_csv('data/disp.csv', na_values=['NA'], sep=';', low_memory=False)
district_data = pd.read_csv('data/district.csv', na_values=['NA'], sep=';', low_memory=False)
card_test_data = pd.read_csv('data/card_test.csv', na_values=['NA'], sep=';', low_memory=False)
loan_test_data = pd.read_csv('data/loan_test.csv', na_values=['NA'], sep=';', low_memory=False)
trans_test_data = pd.read_csv('data/trans_test.csv', na_values=['NA'], sep=';', low_memory=False)
good = pd.read_csv('good.csv', na_values=['NA'], sep=',', low_memory=False)


# Cleaning the data

loan_test_data = loan_test_data.rename(columns={"amount":"loan_amount"})
client_data = client_data.rename(columns={"type":"client_type"})
trans_test_data = trans_test_data.rename(columns={"type":"trans_type", "amount":"trans_amount", "date":"trans_date"})


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
test_data = test_data.drop(columns=["district_id"])
test_data = test_data.merge(client_data, on=['account_id'], how='inner')
test_data = test_data.merge(district_data, left_on='district_id', right_on="code ", how='inner')
test_data = test_data.merge(trans_test_data, on='account_id', how="inner")

test_data = test_data[test_data.std(axis=1) > 0]

test_data['nr_movements'] = 0
test_data['min_trans_amount'] = 0
test_data['max_trans_amount'] = 0
test_data['avg_trans_amount'] = 0
test_data['range_amount'] = 0
test_data['min_trans_balance'] = 0
test_data['max_trans_balance'] = 0
test_data['avg_trans_balance'] = 0
test_data['range_balance'] = 0
test_data['amount_month'] = 0
test_data['withdrawals'] = 0
test_data['credits'] = 0
test_data['members'] = 0
test_data['owner_age'] = 0
test_data['trans_date_diff'] = 0
test_data['able_to_pay'] = False
test_data['balance_last'] = 0
test_data['amount_last'] = 0


from datetime import datetime

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

for index, row in test_data.iterrows():
    trans_rows = test_data[test_data['loan_id'] == test_data.loc[index,'loan_id']]
    test_data.loc[index, 'nr_movements'] = len(trans_rows)
    test_data.loc[index, 'min_trans_amount'] = min(trans_rows['trans_amount'])
    test_data.loc[index, 'max_trans_amount'] = max(trans_rows['trans_amount'])
    test_data.loc[index, 'avg_trans_amount'] = trans_rows['trans_amount'].mean()
    test_data.loc[index, 'range_amount'] = test_data.loc[index, 'max_trans_amount'] - test_data.loc[index, 'min_trans_amount']
    
    test_data.loc[index, 'min_trans_balance'] = min(trans_rows['balance'])
    test_data.loc[index, 'max_trans_balance'] = max(trans_rows['balance'])
    test_data.loc[index, 'avg_trans_balance'] = trans_rows['balance'].mean()
    test_data.loc[index, 'range_balance'] = test_data.loc[index, 'max_trans_balance'] - test_data.loc[index, 'min_trans_balance']
    test_data.loc[index, 'withdrawals'] = len(trans_rows[trans_rows['trans_type'] == 'withdrawal']) + len(trans_rows[trans_rows['trans_type'] == 'withdrawal in cash'])
    test_data.loc[index, 'credits'] = len(trans_rows[trans_rows['trans_type'] == 'credit'])
    test_data.loc[index, 'members'] = len(trans_rows['type'].unique())
    
    max_date = max(trans_rows['trans_date'])
    min_date = min(trans_rows['trans_date'])

    if test_data.loc[index, 'trans_date'] == min_date:
        test_data.loc[index, 'balance_last'] = test_data.loc[index, 'balance']
        test_data.loc[index, 'amount_last'] = test_data.loc[index, 'trans_amount']
    
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
    
    test_data.loc[index, 'trans_date_diff'] = days_between(str(max_year) + "-" + str(max_month) + "-" + str(max_day), str(min_year) + "-" + str(min_month) + "-" + str(min_day))
    test_data.loc[index, 'amount_month'] = sum(trans_rows['trans_amount']) / ((test_data.loc[index, 'trans_date_diff']) / 30.0)
    
    if test_data.loc[index, 'amount_month'] > test_data.loc[index, 'payments']:
        test_data.loc[index, 'able_to_pay'] = True
        
    
    

test_data = test_data.sort_values('loan_id')

test_data = test_data.drop(columns=["account_id", "client_id", "birth_number", "district_id", "disp_id",
                                      "trans_id", "trans_date", "k_symbol", "account", "bank"])

test_data = test_data[test_data['type'] == 'OWNER']

test_data = test_data.drop_duplicates(subset=['loan_id'], keep='first')

test_data = test_data.drop(columns=['trans_type', 'operation',
                          'name ', 'region', 'type'])

test_data = pd.get_dummies(test_data, columns=['frequency'], dtype=bool)
test_data = pd.get_dummies(test_data, columns=['gender'], dtype=bool)
    
tf = test_data[["unemploymant rate '95 ", "no. of commited crimes '95 "]]
tf["unemploymant rate '95 "] = pd.to_numeric(tf["unemploymant rate '95 "], downcast="float", errors='coerce')
tf["no. of commited crimes '95 "] = pd.to_numeric(tf["no. of commited crimes '95 "], downcast="float", errors='coerce')
tf = tf.interpolate()

test_data["unemploymant rate '95 "] = tf["unemploymant rate '95 "]
test_data["no. of commited crimes '95 "] = tf["no. of commited crimes '95 "]

test_data = test_data.drop(columns=['loan_day'])

for index, row in test_data.iterrows():
    test_data.loc[index, 'owner_age'] = test_data.loc[index, 'loan_year'] - test_data.loc[index, 'birth_year']
    
test_data = test_data.drop(columns=["birth_year", "birth_month", "birth_day", "code ",
                        "creation_year", "creation_month", "creation_day",
                        "trans_amount", "balance"])

test_data.to_csv('test.csv', index = False)
