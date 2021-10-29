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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sns.set()

#TODO DO NOT FORGET TO PREPROC EVALUATION SET!!!!!, preProcDf AND preProc Matrix

# -------------------------------------------------

def fetch_train_dataset(path, details=False):
    pd.set_option('display.max_rows', None)

    df = pd.read_csv(path)

    df.rename( columns={'Unnamed: 0':'index'}, inplace=True)
    #remove rows without resonable data
    df= df[pd.to_numeric(df['index'], errors='coerce').notnull()]
    # remove rows without labels
    if(details==True):
        print("-----------------Dropping Rows--------------------\n"+ str(df[df.isnull().any(axis=1)]))
    df.dropna(inplace = True)
    # set numeric columns with missing data to nan
    data_types=['numeric','numeric', 'numeric', 'numeric','numeric', 'numeric', 'string', 'numeric', 'numeric', 'numeric', 'numeric', 'bool', 'numeric']
    #check if there is data in numeric columns of wrong type
    for i in range(0,13):
        if(data_types[i]=='numeric'):
            df["x"+str(i+1)]=pd.to_numeric(df["x"+str(i+1)], errors='coerce')
    # remove outliers
    df.drop(55, inplace=True)
    df.drop(851, inplace=True)

    if details==True:
        print (df)

    df, x7_enc, x12_enc = preProcDf(df)
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
    X=data[:,2:]

    if details==True:
        for i in range(len(X[0])):
            print ("Variance column "+ str(i) + ": " + str(variance(X[:,i])))
    return X,y


def testClassifier(classifier, train_size=0.7, ntrials=100):

    X,y = fetch_train_dataset("data/TrainOnMe-2.csv", details=False)

    means = np.zeros(ntrials,)
    f1_scores= np.zeros(ntrials,)
    X=preProcNp(X)

    # replace nan values with mean
    tr_col_mean=np.nanmean(X, axis=0)
    ids= np.where(np.isnan(X))
    if (len(ids)!=0):
        X[ids] = np.take(tr_col_mean, ids[1])

    for trial in range(ntrials):
        if train_size!=1.0:
            xTr, xTe, yTr, yTe= train_test_split(X,y,train_size=train_size)#,random_state=42
        else:
            xTr= X
            yTr= y
            xTe=[]
            yTe=[]

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        if(len(yTe)!=0):
            yPr = trained_classifier.classify(xTe)

            # Compute classification error
            if trial % 10 == 0:
                print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

                # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(yTe, yPr), display_labels=trained_classifier.classifier.classes_)
                # disp.plot()
                # plt.show()
            means[trial] = 100*np.mean((yPr==yTe).astype(float))
            f1_scores[trial]= f1_score(yTe, yPr, average='micro')

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))
    print("Average F1 Score: "+ str(np.mean(f1_scores)))

    return trained_classifier, np.mean(means), np.std(means)


def preProcDf(df):

    x7_enc = preprocessing.LabelEncoder()
    x7_enc.fit(df['x7'])
    df['x7']=x7_enc.transform(df['x7'])

    x12_enc = preprocessing.LabelEncoder()
    x12_enc.fit(df['x12'])
    df['x12']=x12_enc.transform(df['x12'])

    return df, x7_enc, x12_enc

def preProcNp(X):
    X = X.astype('float64')
    return X

# -------------------------------------------------------------------------

class RandomForestClassifier(object):
    def __init__(self):
        self.trained = False

    def searchHyperparameters(self, XTr, yTr):

        model = Pipeline([
            ('classification', ensemble.RandomForestClassifier(class_weight='balanced', n_estimators=1000))
        ])

        parameter_space = {
            'classification__max_depth': [None,6,15,30,200],
            'classification__min_samples_split': [2,20,50,200],
            'classification__max_leaf_nodes': [None, 5, 20,100],
            'classification__min_samples_leaf': [1, 20,50,100],
            'classification__max_samples': [None, 0.2, 0.5, 0.7],
            'classification__max_features': [None, 'sqrt', 0.5, 0.6, 0.7 ,0.8,0.9]
        }

        # define evaluation
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define search
        #clf = GridSearchCV(estimator=model, param_grid=parameter_space, n_jobs=-1, verbose=1, cv=5, scoring="accuracy")
        clf=RandomizedSearchCV(model, param_distributions=parameter_space, n_iter=500, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
        clf.fit(XTr, yTr)

        # Best parameter set
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    def trainClassifier(self, Xtr, yTr):
        rtn = RandomForestClassifier()
        rtn.classifier =ensemble.RandomForestClassifier(n_estimators=1000, max_depth=Xtr.shape[1]/2+1 ,max_features=0.5, class_weight='balanced')
        rtn.classifier.fit(Xtr, yTr)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


def evaluateData(classifier):
    df = pd.read_csv("data/EvaluateOnMe-2.csv")
    df, x7_enc, x12_enc = preProcDf(df)
    X = df.to_numpy()
    X = preProcNp(X)
    X = X[:, 1:]
    # Train
    trained_classifier = testClassifier(RandomForestClassifier(), train_size=1.0, ntrials=1)[0]
    # Predict
    yPr = trained_classifier.classify(X)
    print(yPr)
    np.savetxt("filename", yPr, newline="", fmt="%s")
    return yPr, trained_classifier


evaluateData(RandomForestClassifier())
testClassifier(RandomForestClassifier(),train_size=0.7, ntrials=100)

#testClassifier(RandomForestClassifier(), train_size=1.0, plot=False, ntrials=1)






