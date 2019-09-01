
# coding: utf-8



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import glob, os
import errno
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.svm import SVC 
from sklearn import svm
from sklearn.model_selection import GridSearchCV

path = r'D:\ml\BAtestdata'
files = glob.glob(os.path.join(path, "*.csv"))
df_ba = []
nr_files = len(files)
print(nr_files)

for i in range(len(files)):
        file =  files[i]
        df_ba.append(pd.read_csv(file,index_col=0))
print(files)
collected_data = pd.read_csv('DatasetThesis.csv',index_col=0)



#Replace the missing values with the median
median = collected_data["Nr. of Competitors"].median()
collected_data["Nr. of Competitors"].fillna(median,inplace=True)
for i in range(nr_files):
    df_ba[i]["Nr. of Competitors"].fillna(median,inplace=True)
    print(df_ba[i])


#Replace the missing values with the mean
mean = collected_data["Revenue(millions of dollars)"].mean()
collected_data["Revenue(millions of dollars)"].fillna(mean,inplace=True)



#Heatmap of the correlations between all the variable
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
sns.heatmap(collected_data.corr(),annot=True, fmt=".2f")


#Normalizing the data and dropping some of the columns
target = collected_data['Acquired'].values
sc_X = StandardScaler()
X_train = collected_data.drop(columns=["Acquired","Nr. of articles","Rounds of seeding ","Revenue(millions of dollars)",
                                       "Nr.of employees"],axis=0)
X_train = sc_X.fit_transform(X_train)
ba_target = []
for i in range(len(files)):
    ba_target.append(df_ba[i]["Acquired"].values)
    df_ba[i] = df_ba[i].drop(columns=["Acquired","Nr. of articles","Rounds of seeding ","Revenue(millions of dollars)",
                                      "Nr.of employees"],axis=0)


y_pred=[]
clf_svc = svm.SVC()
clf_rf = RandomForestClassifier()
lr = LogisticRegression()
best_models=[]
models = [
          clf_rf, 
          lr,
          clf_svc,
          ]

for i, model in enumerate(models):
# finding the best hyperparameters for each model using gridsearch
    if(model==clf_rf):
        grid_search = GridSearchCV(model, param_grid={ 'bootstrap': [True, False],
            'criterion': ['gini', 'entropy'], 'n_estimators': [3,10,30,100,300,1000]}, 
            cv=10, scoring='accuracy', return_train_score=True,n_jobs=-1)
        grid_search.fit(X_train,target)
        best_params= grid_search.best_params_
        best_rf = RandomForestClassifier(bootstrap = best_params['bootstrap'],criterion=best_params['criterion'],
                                         n_estimators= best_params['n_estimators'])
        print(best_params)
        best_models.append(best_rf)
    if(model==lr):
        grid_search = GridSearchCV(model, param_grid={ 'penalty':["l1","l2"], 'C': [0.001, 0.01, 0.1, 1, 10]}
            , cv=10, scoring='accuracy', return_train_score=True,n_jobs=-1)
        grid_search.fit(X_train,target)
        best_params= grid_search.best_params_
        best_lr = LogisticRegression(penalty= best_params['penalty'], C=best_params['C']) 
        print(best_params)
        best_models.append(best_lr)
    if(model==clf_svc):
        grid_search = GridSearchCV(model, param_grid={  'decision_function_shape':('ovo','ovr'),
            'shrinking':(True,False),'kernel':('linear', 'rbf','poly'), 'C': [0.001, 0.01, 0.1, 1, 10], 
            'gamma' : [0.001, 0.01, 0.1, 1]}, cv=10, scoring='accuracy', return_train_score=True,n_jobs=-1)
        grid_search.fit(X_train,target)
        best_params = grid_search.best_params_
        best_svc = svm.SVC(decision_function_shape=best_params['decision_function_shape'],
            shrinking= best_params['shrinking'],kernel=best_params['kernel'], C= best_params['C'], 
            gamma = best_params['gamma'])
        print(best_params)
        best_models.append(best_svc)

kfold = KFold(n_splits=10, random_state=1)
models_table = pd.DataFrame(columns=['Classifier_name', 'train_score', 'vald_score',"ba0_test","ba1_test",
                                     "ba2_test","ba3_test",'ba4_test'])
metrics_table = pd.DataFrame(columns=['Classifier_name', "precision0","recall0","precision1","recall1",
                                      "precision2","recall2","precision3","recall3","precision4","recall4",])
vald_table = pd.DataFrame(columns=['Classifier_name',"vald_precision","vald_recall"])


for i, model in enumerate(best_models):
# training the models 
    print(model)
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore')
    cv_result = cross_validate(model, X_train, target, cv=kfold, scoring='accuracy',return_train_score=True)
    model.fit( X_train, target)
    models_table.loc[i, 'Classifier_name'] = model.__class__.__name__
    models_table.loc[i, 'train_score'] = cv_result['train_score'].mean()
    models_table.loc[i, 'vald_score'] = cv_result['test_score'].mean()
    metrics_table.loc[i, 'Classifier_name']= model.__class__.__name__
    for d in range(len(files)):
# evaluating the trained models on testset,finding the precision and recall of the models 
        y_pred.append(model.predict(df_ba[d]))
        models_table.loc[i,"ba"+str(d)+"_test"] = accuracy_score(ba_target[d],y_pred[d])
        metrics_table.loc[i,'precision'+str(d)] = precision_score(ba_target[d],y_pred[d])
        metrics_table.loc[i, 'recall'+str(d)] = recall_score(ba_target[d],y_pred[d])

    y_pred=[]
    
    y_pred_val = cross_val_predict(model, X_train, target, cv=10)
    print(model.__class__.__name__)
    print(confusion_matrix( target,y_pred_val))
    print()
    vald_table.loc[i,"Classifier_name"] = model.__class__.__name__
    vald_table.loc[i,"vald_precision"] = precision_score(target,y_pred_val)
    vald_table.loc[i,"vald_recall"] = recall_score(target,y_pred_val)

models_table

metrics_table

vald_table

