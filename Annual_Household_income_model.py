# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 17:50:22 2018

@author: Tejinder Wadhwa
"""

import pandas as pd
import numpy as np

#Reading file

adult_df=pd.read_csv('adult_data.csv',header = None, delimiter=' *, *', engine='python')

#%%#
pd.set_option('display.max_columns', None)
adult_df.head()
#%%#
adult_df.shape

#%%#
# Adding column names to the data frame

adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
'marital_status', 'occupation', 'relationship',
'race', 'sex', 'capital_gain', 'capital_loss',
'hours_per_week', 'native_country', 'income']
adult_df.head()
#%%#
# Checking for Missing Data
adult_df.isnull().sum()

# Handling "?" in the data frame
adult_df=adult_df.replace(['?'], np.nan) # replace "?" with NAN

# Again Checking for Missing Data
adult_df.isnull().sum()
#%%#
adult_df_rev = pd.DataFrame.copy(adult_df) # copy of data frame. Direct assignment will create alias
adult_df_rev.describe(include = 'all')
#%%#

#replace missing values with mode values for each categorical variables

for value in ['workclass', 'occupation', 'native_country']:
    adult_df_rev[value].fillna(adult_df_rev[value].mode()[0], inplace=True)

adult_df_rev.isnull().sum()


# creating a list of all categorical variables

colname = ['workclass', 'education', 'marital_status', 'occupation', 'relationship',
'race', 'sex', 'native_country', 'income']

colname

#%%#
#Pre-processing the data
#Converting all Categorical columns to Numeric or multi choice

from sklearn import preprocessing

#%%#
for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    adult_df_rev[x] = le[x].fit_transform(adult_df_rev[x])
#%%#

adult_df.head()
#%%#
adult_df_rev.head()
adult_df_rev.dtypes

#%%#
#OUTLIER DETECTION AND IMPUTED

import matplotlib.pyplot as plt
%matplotlib inline

adult_df.boxplot() #for plotting boxplots for all numerical variables

#%%#

adult_df.boxplot(column='fnlwgt')
plt.show()
#%%#
adult_df.boxplot(column='age')
plt.show()
#%%#
# imputing outliers in Age

#for value in colname:
q1 = adult_df['age'].quantile(0.25) #first quartile value
q3 = adult_df['age'].quantile(0.75) # third quartile value
iqr = q3-q1 #Interquartile range
low = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
print(low)
print(high)

#%%# IMPUTATION OF OUTLIERS IN VARIABLE AGE
 
adult_df_include = adult_df.loc[(adult_df['age'] >= low) & \
(adult_df['age'] <= high)] # meeting the acceptable range
adult_df_exclude = adult_df.loc[(adult_df['age'] < low) | \
(adult_df['age'] > high)] #not meeting the acceptable range

#%%#
print(adult_df_include.shape)
print(adult_df_exclude.shape)

#%%#
#getting back to original shape of df

adult_df_rev=pd.concat([adult_df_include, adult_df_exclude]) #concatenating two dfs
adult_df_rev.shape

#%%#
#Scaling all Numerical variables of X data using StandardScaler() that converts to -3 to +3

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
print(X)

Y.dtype
Y=Y.astype(int) # converting Y column data type integer



#Split data file into test and train

from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
print(Y_train)
Y_train.dtype

#%%# RUNNING A BASIC Logistic Regression MODEL
from sklearn.linear_model import LogisticRegression

#create a model
classifier = LogisticRegression()
#fitting and training the data
classifier.fit(X_train, Y_train)

#%%#

#predicting test data on based of model

Y_pred = classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

#%%#
#Model evaluation

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, recall_score, precision_score

cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")
print(classification_report(Y_test, Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model is :", acc)
#%%#


###########################################
#%%#

#store the predicted probabilities from Logistic Regression Model

y_pred_prob = classifier.predict_proba(X_test)
print(y_pred_prob)

# code to find the best threshold with Best accuracy and least Total & Type 2 Errors
df_list=[]

for a in np.arange(0,1,0.05):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error :", cfm[0,1])
##0.45 is the best threshold for overall errors
    
for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
 #  print(classification_report(Y_test, predict_mine))
    accuracy=accuracy_score(Y_test, predict_mine)
    recall_scr=recall_score(Y_test, predict_mine)
    precision_scr=precision_score(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
          cfm[1,0]," , type 1 error :", cfm[0,1])
    values=[a, total_err, cfm[0,1], cfm[1,0],accuracy,recall_scr,precision_scr]
    df_list.append(values)
    
##0.46 is the best threshold for overall errors

#%%#
print(df_list)
df = pd.DataFrame(df_list, columns=["Precision Val", "Total Err", "Type 1 Err",
                                    "Type 2 Err","Accuracy","T1 Recall","T1 Precision"])
print(df)
print(min(df["Total Err"]))
print(min(df["Type 2 Err"]))
#df.sort_values(['Total Err','Type 2 Err'], ascending=True)
df=df.sort_values(['Total Err', 'Type 2 Err'], ascending=True)
df.to_excel('adult_log_mod.xlsx',index=True,header=True)

# Adjusting the threshold and doing prediction again
y_pred_class=[]

for value in y_pred_prob[:,1]:
    if value > 0.46:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

print(y_pred_class)

cfm2=confusion_matrix(Y_test, y_pred_class)
print(cfm2)

print("Classification report: ")
print(classification_report(Y_test, y_pred_class))

acc2=accuracy_score(Y_test, y_pred_class)
print("Accuracy of the model is :", acc2)

#%%# PLOTTING ROC Curve

from sklearn import metrics

fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_class)
auc = metrics.auc(fpr,tpr)
## auc = 0.5 is worst model, 0.5-0.6 = poor, 0.6-0.7 - Bad, 0.7-0.8 = Good
## auc = 0.8-0.9 = Very Good, 0.9-1.0 = Excellent
print(auc)
print(fpr)
print(tpr)

#%%# plotting roc curve

import matplotlib.pyplot as plt
%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#%%#
from sklearn import metrics
# best auc value from 100 threshold results
fpr, tpr, z = metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

#%%# plotting roc curve

import matplotlib.pyplot as plt
%matplotlib inline
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# K- fold cross validation model
classifier=(LogisticRegression())

from sklearn import cross_validation

#performing kfold_cross_validation
kfold_cv=cross_validation.KFold(n=len(X_train),n_folds=10)
print(kfold_cv)

#running the model using scoring metric as accuracy
kfold_cv_result=cross_validation.cross_val_score(estimator=classifier,X=X_train,
y=Y_train, cv=kfold_cv)
print(kfold_cv_result)

#finding the mean
print(kfold_cv_result.mean())

#Below we are doing logoistic on each kfold data
#predict basis on k-fold data

for train_value, test_value in kfold_cv:
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))

cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")
print(classification_report(Y_test, Y_pred))

acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model is :", acc)
#%%#

####### FEATURE SELECTION TECHNIQUE
#Using Recursive Feature Elimination to find best features, create model and do predictions

colname=adult_df_rev.columns[:]
 
from sklearn.feature_selection import RFE
rfe = RFE(classifier, 10) #retain N variables at the end do logistic regression
model_rfe = rfe.fit(X_train, Y_train)
print("Num Features: ",model_rfe.n_features_) # features retained
print("Selected Features: ") 
print(list(zip(colname, model_rfe.support_))) #true means column retained False dropped
print("Feature Ranking: ", model_rfe.ranking_) # value 1 are indication cols retained. Hi value means dropped first at iteraion and so on

#%%#

Y_pred = model_rfe.predict(X_test)

#%%#
cfm2=confusion_matrix(Y_test, Y_pred)
print(cfm2)

print("Classification report: ")
print(classification_report(Y_test, Y_pred))

acc4=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model is :", acc4)

###################### Code for Model Ensembling

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#print(Y_pred)

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))

