import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#spliting positive and negative into dataframes
def defining(data):
    positive = pd.DataFrame()
    negative = pd.DataFrame()
    y = data['Border']
    for j in range(data.shape[0]):
        if (y[j] == "positive"):
            positive = positive.append(data.iloc[j])
        else:
            negative = negative.append(data.iloc[j])
    return positive,negative


def checkingWithTest(Probability,row):
        true = 0
        false = 0
        #checking to find out who has a greater Probability
        if Probability["pos"] > Probability["neg"]:
            Cvariable = "positive"
        else:
            Cvariable = "negative"
        #matching and appending predictions   
        if (row[1]['Border'] == Cvariable):
            true = true + 1
            predictions.append("Predicted")
        else:
            false= false +1
            predictions.append("Not Predicted")
        return true, false
    
def Prior(size):
    # adding m-estmate
    #addding 1 so that it doesnot becomem zero and 2 for two class variables
    return ((float(len(size)+1))/(float(len(train))+2))

def likely(CVariable,y):
    #finds the likelyhood for each
    #appending the likelyvalue for each ieteration of 
    likelyValue = CVariable[y].value_counts()
    likely = likelyValue.to_dict()
    likely.update(calculatinglikely(likely,CVariable))
    return likely
      
def training(train):

    for y in Columns:
#       temp= [y,likely(positive,y)]
#       P.append(temp) 
        P[y] = [likely(positive,y)]
        N[y] = [likely(negative,y)]
#       temp= [y,likely(negative,y)]
#       N.append(temp)
    likelihood["pos"] = P
    likelihood["neg"] = N
    
    return likelihood

def findinglikelyProb(row,prob):
        for k in likelihood:
            for col in Columns:
                likely = likelihood[k][col]
               #col = "Measure"
                Colum = row[1][col]
                #colum is the test set row
                if Colum in likely:
                    #finding the likelihood for the all feature variables
                    prob = likely[Colum]
                #multiplying all the positives and negative outcomes
            if(k == 'positive'):
                prob = prob * prior[0][0]    
            else:
                prob = prob * prior[1][0]
                
            Probability[k] = prob  
            
        return Probability
    
def testing(likelihood, test):
    prob = 1
    for row in test.iterrows():
        Probability = findinglikelyProb(row,prob)
        true, false = checkingWithTest(Probability,row)
        
    return true, false

def calculatinglikely(likely,CVariable):
    return ((i, float(j+1)/float(len(CVariable)+len(i))) for i, j in likely.items())
 
prior,predictions = [],[]
likelihood,Probability,P,N = {}, {},{},{}
Columns = ['Measure','Port Name','State']
positive,negative = defining(train)
p = Prior(positive),"positive"
prior.append(p)
n = Prior(negative),"negative"
prior.append(n)

likelihood = training(train)


TP, FP = testing(likelihood,test)

test['predictions'] = predictions
test.to_csv('predictions.csv')
#truth and test results
FN = positive.shape[0] - TP
TN = negative.shape[0] - FP

#sensitivity
print("sensitivity = ",TP/(TP+FN))

#specificity
print("specificity = ",TN/(TN+FP))

#accuracy
print("accuracy = ",(TP/test.shape[0])*100)



