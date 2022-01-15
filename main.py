import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
'''

def main():
    train = pd.read_csv('dataTopicF/train_FD001.csv', sep=';')
    test = pd.read_csv('dataTopicF/test_FD001.csv', sep=';')
    cols = train.select_dtypes(include=[float, int]).columns
    train_max = train.select_dtypes(include=[float, int]).max()
    train_min = train.select_dtypes(include=[float, int]).min()
    variance = (train_max - train_min)


    dont_use_cols = [x for x in cols if variance[x] == 0]
    train = train.drop(columns=dont_use_cols)

    test = test.drop(columns=dont_use_cols)

    train.drop_duplicates(keep=False, inplace=True)


    X_train = train.iloc[:, :train.shape[1]-1]
    Y_train = train.iloc[:, train.shape[1]-1]
    X_test = test.iloc[:, :test.shape[1]-1]
    Y_test = test.iloc[:, test.shape[1]-1]




    pca = PCA(n_components=2)
    X_train = (pca.fit_transform(X_train))



    X_test = (pca.transform(X_test))



    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    #grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

    # fitting the model for grid search
    #grid.fit(X_train, Y_train)





    #pred_labels = grid.predict(X_test)
    #print(grid.best_params_)
   # print(grid.score(X_test, Y_test))

    clf = svm.SVC(kernel='rbf', C=1,gamma=0.01, random_state=42)
    clf.fit(X_train,Y_train)

    pred_labels = clf.predict(X_test)
    #{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}
    print(f"Accuracy: {(accuracy_score(Y_test, pred_labels) * 100)}%")

    print(f"Missing Values: {train.isna().sum().sum()}")

    cm = confusion_matrix(Y_test, pred_labels, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
    disp.plot()

    plt.show()
    plot_confusion_matrix(clf, X_test, Y_test)
'''


def main():
    train = pd.read_csv('dataTopicF/train_FD001.csv', sep=';')
    test = pd.read_csv('dataTopicF/test_FD001.csv', sep=';')

    cols = train.select_dtypes(include=[float, int]).columns

    train_max = train.select_dtypes(include=[float, int]).max()
    train_min = train.select_dtypes(include=[float, int]).min()
    variance = (train_max - train_min)


    dont_use_cols = [x for x in cols if variance[x] == 0]
    train = train.drop(columns=dont_use_cols)
    test = test.drop(columns=dont_use_cols)
    train.drop_duplicates(keep=False, inplace=True)



    yes_train = train.loc[train['Failure_status'] == 'yes']
    no_train = train.loc[train['Failure_status'] == 'no']
    yes_test = test.loc[test['Failure_status'] == 'yes']
    no_test = test.loc[test['Failure_status'] == 'no']


    yes = pd.concat([yes_train, yes_test])
    no = pd.concat([no_train, no_test])

    _sample_size = int((yes.shape[0]*4)/5)

    Yes_train = resample(yes, n_samples = _sample_size)
    No_train = resample(no, n_samples = _sample_size)
    train = pd.concat([No_train, Yes_train])

    _temp = no.merge(No_train.drop_duplicates(),
                       how='left', indicator=True)
    _temp = _temp.loc[_temp['_merge']=='left_only']
    No_test = _temp.iloc[:, :_temp.shape[1] - 1]

    _temp = yes.merge(Yes_train.drop_duplicates(),
                       how='left', indicator=True)
    _temp = (_temp.loc[_temp['_merge']=='left_only'])
    Yes_test = _temp.iloc[:, :_temp.shape[1]-1]

    test = pd.concat([No_test, Yes_test])

    X_train = train.iloc[:, :train.shape[1]-1]
    Y_train = train.iloc[:, train.shape[1]-1]
    X_test = test.iloc[:, :test.shape[1]-1]
    Y_test = test.iloc[:, test.shape[1]-1]





    clf = svm.SVC(kernel='rbf', C=1, gamma=0.01, random_state=42)
    clf.fit(X_train, Y_train)
    pred_labels = clf.predict(X_test)
    print(clf.score(X_test, Y_test))
    # {'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}

    print(f"Missing Values: {train.isna().sum().sum()}")

    cm = confusion_matrix(Y_test, pred_labels, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()

    plt.show()
    plot_confusion_matrix(clf, X_test, Y_test)




if __name__ == '__main__':
    main()