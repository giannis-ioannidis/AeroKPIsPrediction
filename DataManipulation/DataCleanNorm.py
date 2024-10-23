import pandas as pd
import numpy as np

#load dfs
train_df = pd.read_csv('PprTrain.csv')
test_df = pd.read_csv('PprTest.csv')

#search for nas
train_df.isnull().sum().any()
test_df.isnull().sum().any()

#drop nas
train_cleandf = train_df.dropna()
#test_cleandf = test_df.dropna()
for i in test_df.columns[test_df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    test_df[i].fillna(test_df[i].mean(),inplace=True)

#check clean dataframe for nas
train_cleandf.isnull().sum().any()
test_cleandf.isnull().sum().any()

#use train set statistics
train_min = np.min(train_cleandf[train_cleandf.columns[:-2]])
train_max = np.max(train_cleandf[train_cleandf.columns[:-2]])

#normalization
def NormalizeData(data):
    out = (data[data.columns[:-2]] - train_min) / (train_max - train_min)
    out['Cost_Index'] = data['Cost_Index']
    out['Max_Payload'] = data['Max_Payload']
    #out['Flight_ID'] = data['Flight_ID']
    out = out.drop('Unnamed: 0', axis=1)
    return out

#apply
#norm_train_data = NormalizeData(train_cleandf)
norm_test_data = NormalizeData(test_cleandf)
