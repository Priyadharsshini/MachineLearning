
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot import figure
warnings.filterwarnings("ignore")
# For Building Classifier Models
from sklearn.neighbors import KNeighborsClassifier

def helper():
    M1 = pd.read_csv("train.sDAT.csv", header=None, names=['x0','x1'])
    M1['label']= 1

    M1test = pd.read_csv("test.sDAT.csv", header=None, names=['x0','x1'])
    M1test['label']= 1

    M2 = pd.read_csv("train.sNC.csv", header=None, names=['x0','x1'])
    M2['label']= 0

    M2test = pd.read_csv("test.sNC.csv", header=None, names=['x0','x1'])
    M2test['label']= 0


    mtrain = (pd.concat([M1, M2])
       .sort_index())
    mtest = (pd.concat([M1test, M2test])
       .sort_index())

    X_train = mtrain.drop(['label'],axis =1)
    Y_train = mtrain["label"]
    X_test = mtest.drop(['label'],axis =1)
    Y_test = mtest["label"]
    return mtrain, mtest, X_train, Y_train, X_test, Y_test

# QUESTION 1
# Train kNN classifiers using the Euclidean distance metric and setting k = 1, 3, 5, 10, 20, 30, 50, 100, 150, 200 respectively. 
# Generate classification boundary visualization plots similar to Figure 2 for each of the trained classifiers. 
# However, your plots should also contain an overlay of the training and test dataset samples colored based on their true class labels. In the plot title, report the error rates achieved on the training and test datasets respectively. 
# Discuss the classification performance of the classifiers trained for various “k” values in the context of over(under)fitting, bias and variance.
def Q1_results():    
    mtrain, mtest, X_train, Y_train, X_test, Y_test = helper()
    gridPoints = pd.read_csv('./2D_grid_points.csv', header=None, names=['x0','x1'])
    kdistancearray = [1, 3, 5, 10, 20, 30, 50, 100, 150, 200]
    ax_count =[0,1,2,3,4,5,6,7,8,9]
    knnAccuracy = []
    fig, axs = plt.subplots(5, 2,figsize=(15, 15),dpi=100)
    axs = axs.flatten()
    x0 = mtrain.x0
    x1 = mtrain.x1
    x2 = mtest.x0
    x3 = mtest.x1
    meshGrid0, meshGrid1 = np.meshgrid(gridPoints.x0.unique(), gridPoints.x1.unique())
    dots = np.reshape(np.stack((meshGrid0.ravel(),meshGrid1.ravel()),axis=1),(-1,2))
    colors = np.array(['green','blue'])

    for (k, i) in zip(kdistancearray, ax_count):
        knnModel = KNeighborsClassifier(n_neighbors=k,metric='euclidean')
        knnModel.fit(X_train, Y_train)
        test_score = knnModel.score(X_test,Y_test)
        train_score = knnModel.score(X_train, Y_train)
        knnAccuracy.append((k, test_score ,train_score))

        estimatedClassLabel = knnModel.predict(dots)
        axs[i].scatter(dots[:,0], dots[:,1], c=colors[estimatedClassLabel], alpha=0.3, linewidths=0)
        axs[i].contour(gridPoints.x0.unique(), gridPoints.x1.unique(), 
                       np.reshape(estimatedClassLabel,(meshGrid0.shape[0],-1)),
                       levels=1, linewidths=1,
                       colors=[colors[0],colors[1],])
        axs[i].scatter(x0, x1, c=colors[mtrain['label']], zorder=2, linewidths=0.5)
        axs[i].scatter(x2, x3, c=colors[mtest['label']], zorder=2, linewidths=0.5, marker = "P")
        axs[i].set_ylabel("x1")
        axs[i].set_xlabel("x0")
        axs[i].set_title('K='+str(k)+'\n Training Error:'+str(round(1-train_score,3))+',Testing Error:'+str(round(1-test_score,3)))

    df = pd.DataFrame(knnAccuracy, columns=['K','Test Score','Train Score'])
    print(df)
    print('Generating results for Q1...')
    plt.show()

# QUESTION 2
# Select the classifier with the lowest test error rate from the experiment in Question 1. 
# Using the “k” value from this classifier but changing the distance metric to Manhattan distance, train a new classifier. 
# Again, generate the visualization plot for the classification boundary with the training and test dataset samples overlaid and colored based on their true labels. 
# In the plot title report the training and test error rates. Discuss the performance of this classifier in comparison to the classifier with the lowest test error rate from Question 1.
def Q2_results():
    mtrain, mtest, X_train, Y_train, X_test, Y_test = helper()
    gridPoints = pd.read_csv('./2D_grid_points.csv', header=None, names=['x0','x1'])
    kdistancearray = [30]
    knnAccuracy = []
    meshGrid0, meshGrid1 = np.meshgrid(gridPoints.x0.unique(), gridPoints.x1.unique())
    dots = np.reshape(np.stack((meshGrid0.ravel(),meshGrid1.ravel()),axis=1),(-1,2))
    colors = np.array(['green','blue'])
    x0 = mtrain.x0
    x1 = mtrain.x1
    x2 = mtest.x0
    x3 = mtest.x1

    for i in kdistancearray: 
        knnModel = KNeighborsClassifier(n_neighbors=i, metric='manhattan')
        knnModel.fit(X_train, Y_train)
        test_score = knnModel.score(X_test,Y_test)
        train_score = knnModel.score(X_train, Y_train)
        knnAccuracy.append((i, test_score ,train_score))

        estimatedClassLabel = knnModel.predict(dots)
        plt.scatter(dots[:,0], dots[:,1], c=colors[estimatedClassLabel], alpha=0.3, linewidths=0)
        plt.contour(gridPoints.x0.unique(), gridPoints.x1.unique(), 
                       np.reshape(estimatedClassLabel,(meshGrid0.shape[0],-1)),
                       levels=1, linewidths=1,
                       colors=[colors[0],colors[1],])
        plt.scatter(x0, x1, c=colors[mtrain['label']], zorder=2, linewidths=0.5)
        plt.scatter(x2, x3, c=colors[mtest['label']], zorder=2, linewidths=0.5, marker = "P")

        plt.ylabel("x1")
        plt.xlabel("x0")
        plt.title('K=30 \n Training Error:'+str(round(1-train_score,3))+',Testing Error:'+str(round(1-test_score,3)))
        print('Generating results for Q2...')
        plt.show()

# QUESTION 3
# Based on the experiments in Question 1 and Question 2, select the distance metric (i.e., Euclidean or Manhattan) that 
# leads to a lower test error rate. Using this chosen distance metric generate the “Error rate versus Model capacity” plot 
# discussed in Lecture 4, Slide 3. As shown in that plot, parameterize “Model capacity” as “ 1 k ” and explore the parameter space 
# from “0.01” to “1.00”. The “x-axis” must be plotted using the “log-scale” and the training and test rate error curves shown. 
# You need not plot the Bayes Classifier error line (is it possible to plot this using only the information you have?). 
# Discuss the trend of the training and test error rate curves in the context of model capacity, bias and variance. 
# Comment on the over(under)fitting zones in the plot.
def Q3_results():
    mtrain, mtest, X_train, Y_train, X_test, Y_test = helper()
    error_rate = []
    error_rate_train = []
    kdistancearray = [1, 3, 5, 10, 20, 30, 50, 100]
    # new_yticklabels = ["0.01", "0.02", "0.03", "0.05", "0.1", "0.2","0.33","1"]
    for i in kdistancearray: 
        knnModel = KNeighborsClassifier(n_neighbors=i, metric='euclidean')
        knnModel.fit(X_train, Y_train)
        PredictTrain = knnModel.predict(X_train)
        error_rate_train.append(np.mean(PredictTrain != Y_train))

        PredictTest = knnModel.predict(X_test)
        error_rate.append(np.mean(PredictTest != Y_test))

    plt.figure(figsize=(10,6))

    plt.plot(kdistancearray,error_rate_train,color='lightblue', linestyle='dashed', 
             marker='o',markerfacecolor='lightblue', markersize=8)
    plt.plot(kdistancearray,error_rate,color='orange', linestyle='dashed', 
             marker='o',markerfacecolor='orange', markersize=8)
    plt.title('Error Rate vs. K Value')
    plt.xlabel('1/K')
    # As it says it has to be plotted using log scale
    plt.xscale('log')
    plt.legend(['Training', 'Testing'], loc ="center right", borderpad=2)
    plt.ylabel('Error Rate')
    print('Generating results for Q3...')
    plt.show()

def diagnoseDAT(Xtest, data_dir):
    knn_pipe = Pipeline([('mms', MinMaxScaler()),
                     ('knn', KNeighborsClassifier())])
    params = [{'knn__n_neighbors': [30],
         'knn__weights': ['uniform', 'distance'],
         'knn__leaf_size': [15, 20]}]
    gs_knn = GridSearchCV(knn_pipe,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5)
    path = data_dir + "train.sDAT.csv"
    path1 = data_dir + "train.sNC.csv"

    M1 = pd.read_csv(path, header=None, names=['x0','x1'])
    M1['label']= 1
    
    M2 = pd.read_csv(path1, header=None, names=['x0','x1'])
    M2['label']= 0
    
    mtrain = (pd.concat([M1, M2]).sort_index())
    X_train = mtrain.drop(['label'],axis =1)
    Y_train = mtrain["label"]
    
    
    gs_knn.fit(X_train, Y_train)
    print(gs_knn.predict(Xtest))
    return gs_knn.predict(Xtest)

# diagnoseDAT(X_test_No_Outliers,"file:///Users/priyadharsshinis/Downloads/" )


if __name__ == "__main__":  
    # mtrain, mtest, X_train, Y_train, X_test, Y_test = helper()
    Q1_results()
    Q2_results()
    Q3_results()
    # diagnoseDAT(X_test,"file:///Users/priyadharsshinis/Downloads/")


    
