
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data_0riginal = pd.read_csv("Q3.csv")

def kfold(data,X,y,i,accuracyTotal):
    #spliting positive and negative into dataframe
    positive = pd.DataFrame()
    negative = pd.DataFrame()
    for j in range(data.shape[0]):
        if (y[j] == 1):
            positive = positive.append(data.iloc[j])
        else:
            negative = negative.append(data.iloc[j])
    
    #taking length for test and train
    lp = int(positive.shape[0]/5)
    ln = int(negative.shape[0]/5)
    fold = []
    stp = 0 #starting positive
    stn = 0 #starting negative
    
    for j in range(5):
       # p = len(positive) - lp
        p = pd.DataFrame()
        n = pd.DataFrame()
        for k in range(stp,lp):
            p = p.append(positive.iloc[k])
            
        for k in range(stn,ln):
            n = n.append(negative.iloc[k])
            
        f = pd.DataFrame()
        f = f.append(p)
        f = f.append(n)
        #creating each fold stratified
        fold.append(f)
        stp = lp
        lp = lp + int(positive.shape[0]/5)
        stn = ln
        ln = ln + int(negative.shape[0]/5)
        
    accuracy = StatCV(X,y,i,fold)
    a = [np.mean(accuracy),i]
    accuracyTotal.append(a)
    
def shuffling(fold):
    fold = fold.reindex(np.random.permutation(fold.index)) 
    return fold

#does random forest and uses stratified cross validation
def StatCV(X,y,i,fold):
    accuracy = []
    for j in range(5):
        temp_df  = shuffling(fold[j])
        #spliting into  training and testing
        index = int(0.75 * len(temp_df))
        X_train, X_test = X[:index], X[index:]
        y_train, y_test = y[:index], y[index:]   
        accuracy.append(Random(X_train,X_test,y_train,y_test))
    return accuracy
           
#random forest model
def Random(X_train,X_test,Y_train,Y_test): 
    rf = RandomForestRegressor(n_estimators= 100, random_state=42)
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    acc = [100*np.mean(predictions)]
    #accuracy.append(acc)
    return acc
#finding accuracy of the featureSet
def calculateAcc(featureSet):
    data = pd.DataFrame()
    accuracyTotal = []
    for j in range(len(featureSet)):
        a = []
        for i in range(data_0riginal.shape[0]): 
            a.append(data_0riginal.iloc[i][featureSet[j]])
            
        data[featureSet[j]] = a        
    X = data
    y = tempdata['class']
    
    kfold(data,X,y,100,accuracyTotal)
    accuracyTotal.sort(reverse = True)
    return accuracyTotal  
    
featureSet = [] 
tempdata = data_0riginal.copy()

def findBestFeature(tempdata):
    accuracyTotal = []
    for i in range(1,len(tempdata.columns)-1):
        X = tempdata.iloc[ :, i-1:i]
        y = tempdata['class']
        kfold(tempdata,X,y,i,accuracyTotal)
    accuracyTotal.sort(reverse = True)    
    return accuracyTotal
    
#the 1st set of feature
b = findBestFeature(tempdata)
featureSet.append(tempdata.columns[b[0][1]])
del tempdata[featureSet[0]]

acc = []
m = 1
# the set of features without the 1st best feature and deleting the feature from the dataset
for k in range(len(data_0riginal.columns)):    
    b = findBestFeature(tempdata)    
    # calculating accuracy for the feature set 
    #print(calculateAcc(featureSet) , "for", len(featureSet) )
    acc.append(calculateAcc(featureSet))
    
    if (m  > acc[k][0][0]):
        break
    else:
        featureSet.append(tempdata.columns[b[0][1]])
        del tempdata[featureSet[k+1]]
        m = acc[k][0][0]
    
print("feature set is = ", featureSet)


  
