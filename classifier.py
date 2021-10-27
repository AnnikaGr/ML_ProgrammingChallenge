from statistics import variance

import pandas as pd
import numpy as np
from sklearn import decomposition, tree, preprocessing
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn.preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, RandomizedSearchCV
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

sns.set()

# TODO handle outliers, ausreiser in reihe 843, und irgendwo im letzten Label x11-55, x4-851
# TODO scale data
# TODO maybe: research names connections
# TODO research on classifiying mixed data
# TODO AUC

# TODO check overfitting/hyperparametersearch
# TODO https://towardsdatascience.com/boost-machine-learning-performance-by-30-with-normalization-and-standardization-156adfbf215b
# TODO https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a

#TODO is this dataset inbalanced?
# TODO oversampling for atsuto, So we should balance our dataset before training our classifier. There are various methods to do this, including subsampling (taking a smaller yet equal selection of samples from each class), upsampling (taking repeat samples from some classes to increase its numbers), resampling (using an algorithm like SMOTE to augment the dataset with artificial data), etc
# https://medium.com/@b.terryjack/tips-and-tricks-for-multi-class-classification-c184ae1c8ffc

#TODO DO NOT FORGET TO PREPROC EVALUATION SET!!!!!, preProcDf AND preProc Matrix

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
    #check if there is data in numeric columns of wrong type #TODO also for evaluation?
    for i in range(0,13):
        if(data_types[i]=='numeric'):
            df["x"+str(i+1)]=pd.to_numeric(df["x"+str(i+1)], errors='coerce')

    if details==True:
        print (df)

    df= preProcDf(df)
    if details==True:
        for i in range(2, len(df.iloc[0])):
            col= df.iloc[:,i]
            plt.figure();
            col.plot(kind="hist");
            plt.xlabel(df.columns[i])
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.show()

    data = df.to_numpy()
    y=data[:,1]
    X=data[:,2:] # TODO gut wenn nur bis 12 mit hot encoding (alle categorical rauslassen)

    for i in range(len(X[0])):
        tmp=X[:,i]
        print ("Variance column "+ str(i) + ": " + str(variance(X[:,i])))
    return X,y

""" Train-Test split

    Parameters:
    Y: labels
    X: features
    pcSplit: ratio of training and testing data
    seed: inhibit randomness

    Returns:
    

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

    means = np.zeros(ntrials,)
    X=preProcNp(X)

    for trial in range(ntrials):

        #print("TRIAL", trial)

        #xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)
        xTr, xTe, yTr, yTe= train_test_split(X,y,test_size=0.3,random_state=42) #,random_state=42

        # replace nan values with mean
        tr_col_mean=np.nanmean(xTr, axis=0)
        ids= np.where(np.isnan(xTr))
        if (len(ids)!=0):
            xTr[ids] = np.take(tr_col_mean, ids[1])

        te_col_mean=np.nanmean(xTe, axis=0)
        ids= np.where(np.isnan(xTe))
        if(len(ids)!=0):
            xTe[ids]= np.take(te_col_mean, ids[1])

        # oversample minority class
        #print("lenght yTr:" + str(len(xTr)))

        # print('Atsuto: '+ str(len(np.where(yTr=='Atsuto')[0])))
        # print('Jorg: '+ str(len(np.where(yTr=='Jorg')[0])))
        # print('Shoogee: '+ str(len(np.where(yTr=='Shoogee')[0])))
        # print('Bob: '+ str(len(np.where(yTr=='Bob')[0])))
        #print("Before oversampling -----------------------------------------------")
        #classifier.searchHyperparameters(xTr, xTe, yTr, yTe)

        dic={'Atsuto':150}
        #cat_mask=[False, False,False,False,False, False, False,False,False,False, False, True,True,True,True,True]
        oversample = RandomOverSampler(sampling_strategy='not majority')
        xTr, yTr = oversample.fit_resample(xTr, yTr)

        # print('Atsuto: '+ str(len(np.where(yTr=='Atsuto')[0])))
        # print('Jorg: '+ str(len(np.where(yTr=='Jorg')[0])))
        # print('Shoogee: '+ str(len(np.where(yTr=='Shoogee')[0])))
        # print('Bob: '+ str(len(np.where(yTr=='Bob')[0])))


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
    return trained_classifier, np.mean(means), np.std(means)

def plotAccuracy(means):
    indices = np.arange(1, len(means)+1)
    df = pd.DataFrame(np.column_stack((indices,means)), columns = ['Trial','Classification Accuracy'])
    plt.figure()
    sns.lineplot(x='Trial', y='Classification Accuracy', data=df)
    plt.ylim(50, 100)
    plt.show()

""" Preproc String data in feature matrix

    Parameters:
    df: pandas dataframe

    Returns:
    feature matrix with strings substituted by unique integers

"""
def preProcDf(df):

    #handle categorical data
    # x7_dummy= pd.get_dummies(df['x7'])
    # x12_dummy= pd.get_dummies(df['x12'])
    # x12_dummy.describe()
    # x12_dummy.columns = ['False','True']
    # x7_dummy.describe()
    # x7_dummy.columns = ['name1','name2', 'name3', 'name4', 'name5']
    # df=pd.concat((df,x7_dummy,x12_dummy), axis=1)
    # df=df.drop(['x7', 'x12'], axis=1)
    # df=df.drop(['False'], axis=1)
    # df=df.drop(['name5'], axis=1)

    le = preprocessing.LabelEncoder()
    le.fit(df['x7'])
    df['x7']=le.transform(df['x7'])

    le = preprocessing.LabelEncoder()
    le.fit(df['x12'])
    df['x12']=le.transform(df['x12'])

    df.drop(55, inplace=True)
    df.drop(851, inplace=True)

    return df

def preProcNp(X):
    #TODO handle outlier here! RobustScaler? Why is mean wit

    #TODO next line also needed for evaluation data?
    #print("Before: "+ str(X.mean(axis=0)))
    X = X.astype('float64')
    # transformer = StandardScaler() #RobustScaler()
    # transformer.fit(X)
    # X= transformer.transform(X)
    # #X= transformer.fit_transform(X)

    #print("After: "+ str(X.mean(axis=0)))
    return X

def searchHyper(classifier):
    X,Y = fetch_dataset("data/TrainOnMe-2.csv", details=False) ##TODO ,pcadim

    X=preProcNp(X)

    # replace nan values with mean
    tr_col_mean=np.nanmean(X, axis=0)
    ids= np.where(np.isnan(X))
    if (len(ids)!=0):
        X[ids] = np.take(tr_col_mean, ids[1])

    # oversample = RandomOverSampler(sampling_strategy='not majority')
    # X, Y = oversample.fit_resample(X, Y)
    classifier.searchHyperparameters(X, Y)

# -------------------------------------------------------------------------

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

class RandomForestClassifier(object):
    def __init__(self):
        self.trained = False

    def searchHyperparameters(self, XTr, xTe, yTr, yTe):
        rtn = RandomForestClassifier()

        model = Pipeline([
            ('sampling', RandomOverSampler()),
            ('classification', ensemble.RandomForestClassifier())
        ])

        #forest =ensemble.RandomForestClassifier(n_estimators=400, max_depth=50)
        parameter_space = {
            # 'classification__n_estimators': [1000],
            'classification__max_depth': [None,6,15,30,200],
            'classification__min_samples_split': [2,20,50,200],
            # 'classification__max_leaf_nodes': [None, 5, 20,100],
            'classification__min_samples_leaf': [1, 20,50,100],
            'classification__max_samples': [None, 0.2, 0.5, 0.7],
            'classification__max_features': [None, 'sqrt', 0.5, 0.6, 0.7 ,0.8,0.9]
        }

        # parameter_space = {
        #     'classification__n_estimators': [1000],
        #     'classification__max_depth': [None, 4,6,9,11,15,20,30,200],
        #     'classification__min_samples_split': [2,20,50,200],
        #     'classification__max_leaf_nodes': [None, 5,10, 20,100],
        #     'classification__min_samples_leaf': [1, 10,20,50,100,300],
        #     'classification__max_samples': [None, 0.2, 0.5, 0.7],
        # }


        # define evaluation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define search
        #clf = GridSearchCV(estimator=model, param_grid=parameter_space, n_jobs=-1, verbose=1, cv=5, scoring="accuracy")
        clf=RandomizedSearchCV(model, param_distributions=parameter_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
        clf.fit(XTr, yTr)


        # Best paramete set
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        print("Best estimator and their score: ")
        print(clf.best_estimator_)
        print(clf.score(xTe, yTe))


    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = RandomForestClassifier()

        rtn.classifier =ensemble.RandomForestClassifier(n_estimators=1000, max_depth=Xtr.shape[1]/2+1, max_features=0.5)

        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class GradientBoostingClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = GradientBoostingClassifier()
        rtn.classifier = ensemble.GradientBoostingClassifier(n_estimators=100)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)

class NaiveBayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = NaiveBayesClassifier()
        rtn.classifier = GaussianNB()
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class KNeiClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = KNeiClassifier()
        rtn.classifier = KNeighborsClassifier(n_neighbors=3)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class LogisticRegressionClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = LogisticRegressionClassifier()

        Xtr= sklearn.preprocessing.scale(Xtr)
        rtn.classifier = LogisticRegression(max_iter=100000000)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)

class SVMLinClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = SVMLinClassifier()
        rtn.classifier = SVC(probability=True, kernel="linear")
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)

class SVMDefClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = SVMDefClassifier()
        rtn.classifier = SVC(probability=True)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class NNClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        mlp = MLPClassifier(max_iter=10000)
        parameter_space = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }

        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        clf.fit(Xtr, yTr)

        # Best paramete set
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        rtn = NNClassifier()
        rtn.classifier= clf
        #rtn.classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=10000)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


class StockGradDesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = StockGradDesClassifier()
        rtn.classifier = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)



scores=[]

#scores.append(testClassifier(DecisionTreeClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(RandomForestClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(GradientBoostingClassifier(),split=0.7, plot=True))

#scores.append(testClassifier(NaiveBayesClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(KNeiClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(LogisticRegressionClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(SVMLinClassifier(),split=0.7, plot=True))
#scores.append(testClassifier(SVMDefClassifier(),split=0.7, plot=True))


testClassifier(RandomForestClassifier(),split=0.7, plot=False, ntrials=100)
#searchHyper(RandomForestClassifier())

scores=scores




print(scores)

