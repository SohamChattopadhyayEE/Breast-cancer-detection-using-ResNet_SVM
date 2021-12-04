import numpy as np
import pickle as pkl

clf_model = 'utils/svm_clf.pkl'
file_ = open(clf_model, 'rb')
clf_svm = pkl.load(file_)
file_.close()


def predict(feature):
    pred = clf_svm.predict(feature)

    if pred == 1:
        return 'Malignant'
    else:
        return 'Benign'


if __name__=='__main__':
    x = np.random.rand(1,512)
    pred = predict(x)
    print(pred)

