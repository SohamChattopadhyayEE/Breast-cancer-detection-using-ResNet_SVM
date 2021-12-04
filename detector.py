import warnings


import utils.ResNet18_features as rnf
import utils.classification.svm_clf as clf

warnings.filterwarnings("ignore")

def detection(file):
    feature = rnf.get_features(file)
    return clf.predict(feature)

if __name__ == '__main__':
    path = 'Sample_data/Malignant.png'
    prediction = detection(path)
    print(prediction)
