import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

account_data = pd.read_csv('data/account.csv', na_values=['NA'], sep=';', low_memory=False)
client_data = pd.read_csv('data/client.csv', na_values=['NA'], sep=';', low_memory=False)
disp_data = pd.read_csv('data/disp.csv', na_values=['NA'], sep=';', low_memory=False)
district_data = pd.read_csv('data/district.csv', na_values=['NA'], sep=';', low_memory=False)

#train
card_train_data = pd.read_csv('data/card_train.csv', na_values=['NA'], sep=';', low_memory=False)
loan_train_data = pd.read_csv('data/loan_train.csv', na_values=['NA'], sep=';', low_memory=False)
trans_train_data = pd.read_csv('data/trans_train.csv', na_values=['NA'], sep=';', low_memory=False)

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