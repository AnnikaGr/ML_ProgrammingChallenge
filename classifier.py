import pandas as pd
import numpy as np
from sklearn import decomposition, tree
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

"""Fetches training data

    Parameters:
    path (string): path to training data file

    Returns:
    pandas dataframe containing features and labels

"""
def fetch_dataset(path, details=False):
    pd.set_option('display.max_rows', None)
    #pd.set_option('display.max_columns', None)
    df = pd.read_csv (path)
    df.rename( columns={'Unnamed: 0':'index'}, inplace=True)

    #remove rows without resonable data
    df= df[pd.to_numeric(df['index'], errors='coerce').notnull()]

    # remove data without labels
    if(details==True):
        print("-----------------Dropping Rows--------------------\n"+ str(df[df.isnull().any(axis=1)]))
    df.dropna(inplace = True)

    # set numeric columns with missing data to nan
    data_types=['numeric','numeric', 'numeric', 'numeric','numeric', 'numeric', 'string', 'numeric', 'numeric', 'numeric', 'numeric', 'bool', 'numeric']
    #check if there is data in numeric columns of wrong type
    for i in range(0,13):
        if(data_types[i]=='numeric'):
            df["x"+str(i+1)]=pd.to_numeric(df["x"+str(i+1)], errors='coerce')

    if details==True:
        print (df)

    data = df.to_numpy()
    #TODO construct testing and training vectors
    y=data[:,1]
    X=data[:,2:15]
    return X,y

""" Train-Test split

    Parameters:
    Y: labels
    X: features
    ratio: ratio of training and testing data

    Returns:
    np arrays containing training and testing data

"""
# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx


def testClassifier(classifier, dim=0, split=0.7, ntrials=100, plot = False):

    X,y = fetch_dataset("data/TrainOnMe-2.csv", details=False) ##TODO ,pcadim

    pcadim=0

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        #print("TRIAL", trial)

        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            #print("PCA-Components: "+ str(pca.components_))
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    if(plot==True):
        plotAccuracy(means)

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))

def plotAccuracy(means):
    indices = np.arange(1, len(means)+1)
    df = pd.DataFrame(np.column_stack((indices,means)), columns = ['Trial','Classification Accuracy'])
    plt.figure()
    sns.lineplot(x='Trial', y='Classification Accuracy', data=df)
    plt.ylim(50, 100)
    plt.show()


# try out different base classifiers
class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth=Xtr.shape[1]/2+1)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


#TODO scale data?

testClassifier(DecisionTreeClassifier(),split=0.7, plot=True)







