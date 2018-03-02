import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def pre_processing():
    """Function reads in csv file and performs necessary pre-processing steps
    to calculate moving averages for goals/shots etc over the course of the last 5 games."""
    global data
    data = pd.read_csv('book.csv', header = 0) #Read in dataset
    data = data.rename(columns = {'HS Taregt': 'HST'}) #Rename column
    data['index'] = data.index #create index column for sorting
    data = data.sort_values(by = ['HomeTeam', 'index']) #Sort alphabetically
    #Create moving averages
    HG_AV = pd.rolling_mean(data['FTHG'].groupby(data['HomeTeam']), 5)
    HS_AV = pd.rolling_mean(data['Home Shots'].groupby(data['HomeTeam']), 5)
    HST_AV = pd.rolling_mean(data['HST'].groupby(data['HomeTeam']), 5)
    #Add columns in
    data['HG_AV'] = list(HG_AV)
    data['HS_AV'] = list(HS_AV)
    data['HST_AV'] = list(HST_AV)
    #Sort alphabetically by away team
    data = data.sort_values(by = ['AwayTeam', 'index'])
    AG_AV = pd.rolling_mean(data['FTAG'].groupby(data['AwayTeam']), 5)
    AS_AV = pd.rolling_mean(data['Away Shots'].groupby(data['AwayTeam']), 5)
    AST_AV = pd.rolling_mean(data['AST'].groupby(data['AwayTeam']), 5)
    data['AG_AV'] = list(AG_AV)
    data['AS_AV'] = list(AS_AV)
    data['AST_AV'] = list(AST_AV)
    #Sort it back to original
    data = data.sort_values(by = ['index'])
    #Drop any values containing nan
    data = data.dropna()
    #Write file to csv so can check
    data.to_csv('processed_book.csv')
    return data

pre_processing() #call preprocessing function
#Split into X and Y, independent and dependent variables.
X = data[['B365H', 'B365D', 'B365A', 'HG_AV', 'HS_AV', 'HST_AV', 'AG_AV', 'AS_AV', 'AST_AV']]
Y = data[['FTR']]

#Split dataset into train/test.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Convert to numpy arrays
X_train, X_test, y_train = X_train.values, X_test.values, y_train.values
#Create random forest
rf = RandomForestClassifier(n_estimators = 750, oob_score = True)
rf.fit(X_train, y_train)

#Make predictions on test set
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')

#Make confusion matrix to assess accuracy.
cm = pd.DataFrame(confusion_matrix(y_test, predicted), columns = data.FTR.unique(), index = data.FTR.unique())
print(cm)
