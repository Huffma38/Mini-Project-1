import pandas
from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df

# Part 1

scores = []
vidBehaviors = ['fracSpent','fracComp','fracPaused','numPauses','avgPBR','numRWs','numFFs']

dataset_2 = dataset_1.groupby('userID').count().reset_index()
names = dataset_2['userID'][dataset_2['VidID'] > 5].values

dataset_3 = dataset_1[dataset_1['userID'].isin(names)].groupby('userID').mean().reset_index()
tempDF = dataset_3[vidBehaviors].copy()
data = tempDF.values
numKs = range(1,11)
for k in numKs:
    kmeans = cluster.KMeans(n_clusters=k, n_init='auto').fit(data)
    scores.append(kmeans.inertia_)
plt.scatter(numKs, scores)
plt.ylabel('Sum Of Squared Distances')
plt.xlabel('K Number of Clusters')
plt.show()
# plot clearly shows should use 3?? or 4?? kmeans

# Problem 2
# NO IDEA IF THIS IS CORRECT

'''
videos = pandas.unique(dataset_1['VidID'])
for vid in videos:
    thisDf = dataset_1[dataset_1['VidID'] == vid]
    X = thisDf[vidBehaviors]
    y = thisDf['s']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2, random_state=8675309)
    reg = linear_model.Ridge(alpha=0.1, fit_intercept=True)
    '''
X_train, X_test, y_train, y_test = train_test_split(dataset_3[vidBehaviors].values, dataset_3['s'].values, test_size=0.2, random_state=4221)
# Train a linear regression model
print(dataset_3.head())
print(X_train[:10])
print(y_train[:])
model = linear_model.LinearRegression().fit(X_train, y_train)

# Predict performance on the test set
predictions = model.predict(X_test)
print(predictions[:10])
print(y_test[:10])
# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)
print(f'Mean Squared Error: {mse}') 
print(f'R^2 Score : {r2}')