import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn import svm


def acc_cal(pred, lab):
  total = len(lab)
  correct = 0
  for i in range(len(lab)):
    correct += lab[i]==pred[i]
  return correct*100/total

def train_val(data_path, label_path):
    X_df = pd.read_csv(data_path)
    Y_df = pd.read_csv(label_path)

    X = np.array(X_df)
    Y = np.array(Y_df)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    acc = acc_cal(pred, Y_test)
    file_ = open('utils/svm_clf_train_val.pkl', 'wb')
    pkl.dump(clf, file_)
    file_.close()
    print(acc)


def train(data_path, label_path):
    X_df = pd.read_csv(data_path)
    Y_df = pd.read_csv(label_path)

    X = np.array(X_df)
    Y = np.array(Y_df)
    clf_svm = svm.SVC(kernel='rbf')
    clf_svm.fit(X, Y)

    file_ = open('utils/svm_clf.pkl', 'wb')
    pkl.dump(clf_svm, file_)
    file_.close()
    print("SVM classifier model saved successfully")

if __name__=='__main__':
    data_path = 'Dataset/ResNet18_features.csv'
    label_path = 'Dataset/Lables.csv'
    train_val(data_path, label_path)
    train(data_path, label_path)