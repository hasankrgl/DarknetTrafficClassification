
#import os

#import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score



### Constants
SEED = 42 
NUM_FEATURES = 5 
TRAIN_PCT = 0.75 

MAX_DEPTH = 4 
MAX_ITER = 200 
N_NEIGHBORS = 5 



#Reading data from csv
traffic = pd.read_csv("Your CSV FILE")
print(f'Number of Rows: {traffic.shape[0]}') #Rows
print(f'Number of Columns: {traffic.shape[1]}') # Columns
traffic.head() .

#Removing empty columns
traffic.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=False) 
traffic = traffic.loc[:, traffic.apply(pd.Series.nunique) != 1] 

#Function to describe evulation metrics.
def calculateMetrics(y_test, y_pred): 
    acc = accuracy_score(y_test, y_pred) 
    recall = recall_score(y_test, y_pred, average="macro") 
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    mse = mean_squared_error(y_test, y_pred) 
    f1score = f1_score(y_pred, y_test, average='weighted') 
    cm=confusion_matrix(y_test, y_pred) 
    print(">>> Metrics")
    print(f'- Accuracy  : {acc}')
    print(f'- Recall    : {recall}')
    print(f'- Precision : {precision}')
    print(f'- MSE       : {mse}')
    print(f'- F1 Score  : {f1score}')
    print("- Confusion Matrix :")
    print(cm)
    
#Function to select best features
def selectFeatures(x, y, train_size_pct=0.75):
    """
    selectFeatures
        x : The data which is going to used to train.
        y : The data which is going to trained.
        train_size_pct :
        @return (list)
    """

    # Feature selection algorithms
    rf = RandomForestClassifier(max_depth=MAX_DEPTH, criterion='entropy', random_state=SEED) 
    dectree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)

    classifier_mapping = {
        "RandomForest" : rf,
        "DecisionTree" : dectree
    }

   
    X_train_fs, X_test_fs, Y_train_fs, Y_test_fs = train_test_split(x, y, train_size=train_size_pct)

    model_features = {}

    for model_name, model in classifier_mapping.items(): 
        print(f'[Training] {model_name}')
        start_train = datetime.now() 
        model.fit(X_train_fs, Y_train_fs) 
        print(">>> Training Time: {}".format(datetime.now() - start_train))
        model_features[model_name] = model.feature_importances_
        model_score = model.score(X_test_fs, Y_test_fs)
        print(f'>>> Training Accuracy : {model_score*100.0}') 
        print("")

    cols = X_train_fs.columns.values 
    feature_df = pd.DataFrame({'features': cols}) 
    for model_name, model in classifier_mapping.items(): 
        feature_df[model_name] = model_features[model_name] 

   
    all_f = []
    for model_name, model in classifier_mapping.items():
        try:
            all_f.append(feature_df.nlargest(NUM_FEATURES, model_name)) 
        except KeyError as e:
            print(f'*** Failed to add features for {model_name} : {e}')
    result = []
    
    for i in range(len(all_f)):
        result.extend(all_f[i]['features'].to_list())		

   
    selected_features = list(set(result))					

    return selected_features #

#Function to describe classification
def train_test_model(model_name, model, x, y, train_size_pct): 

   
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train_size_pct)

    
    print(f'\n[Training] {model_name}')
    start_train = datetime.now() 
    model.fit(X_train, Y_train) 
    print(f'>>> Training time: {datetime.now() - start_train}')

    
    train_acc = model.score(X_train, Y_train) 
    print(f'>>> Training accuracy: {train_acc}')

    
    start_predict = datetime.now() 
    y_pred = model.predict(X_test) 
    print(f'>>> Testing time: {datetime.now() - start_predict}')

   
    calculateMetrics(Y_test, y_pred) 
    
#Function which actually does classification.    
def evaluateIndividualClassifiers(x, y, train_size_pct): 
    """
    evaluateIndividualClassifiers
        x : Tahminler için kullanılacak olan veri kümeleri
        y : X içindeki her satır için hedef sınıf
        train_size_pct : Eğitim aralığı (0.75-0ve1 arasında)
    """

    rf = RandomForestClassifier(max_depth=MAX_DEPTH, random_state=SEED) 
    dectree = DecisionTreeClassifier(max_depth=MAX_DEPTH, criterion='entropy', random_state=SEED) 
    #sc=DecisionTreeClassifier(criterion='gini',max_depth=MAX_DEPTH,random_state=SEED) 
    #knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS) 
    #mlpnn = MLPClassifier(max_iter=MAX_ITER) 
    
    
    classifier_mapping = {
        f'RandomForest-{MAX_DEPTH}' : rf,
        f'DecisionTree-{MAX_DEPTH}' : dectree,
        #f'KNeighbors-{N_NEIGHBORS}' : knn,
        #f'MLP-{MAX_ITER}' : mlpnn,
        #f'SimpleCART-{MAX_DEPTH}':sc,
        
    }
    
    for model_name, model in classifier_mapping.items(): 

        train_test_model(model_name, model, x, y, train_size_pct) 
        
    print('>>> Decision Tree Leaves/Nodes/Depth')
    print('- Tree Depth : ',dectree.tree_.max_depth) 
    print('- No of Leaves : ',dectree.tree_.n_leaves) 
    print('- No of Nodes : ',dectree.tree_.node_count) 
    print('>>> Simple Cart Leaves/Nodes/Depth')
    #print('- Tree Depth : ',sc.tree_.max_depth) 
    #print('- No of Leaves : ',sc.tree_.n_leaves) 
    #print('- No of Nodes : ',sc.tree_.node_count) 
    
    
        
X = traffic.iloc[:, 0:(traffic.shape[1]-1)] 
Y = traffic.iloc[:, -1] 

selected_features = selectFeatures(X, Y) 
print(f'Selected Features "from All": {selected_features}')
Xse_all = X[selected_features] 
print(f'[*] Beginning evaluations: Selected Features (from "All Features")"')
evaluateIndividualClassifiers(Xse_all, Y, TRAIN_PCT) 
