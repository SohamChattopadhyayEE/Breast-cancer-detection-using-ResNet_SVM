# Automatic Breast Cancer Detection From Hystopathology Images

![gif](https://github.com/SohamChattopadhyayEE/Breast-cancer-detection-using-ResNet_SVM/blob/main/videos/Benign.gif)
- This is a Python project for breast cancer detection. The user has to give input of an `histopathological images` of `40x` magnification scale and the model will predict whether it is `cancerous (malignant)` or `non-cancerous (benign)`. 
- This entire end-to-end full-stack api utilizes popular frameworkss like: 
    - Flask
    - Jinja2
    - OpenCV
    - Pytorch
    - Scikit-learn

## Description
- The proposed framework is a hybrid of Deep Learning and traditional Machine Learning.  
- Following traditional `Transfer learning` technique `pre-trained ResNet18` model trained on BreakHis breast-cancer dataset is used for feature extraction. 
- High-dimensional `pre-final layer features` of ResNet18 having dimension of `512` are extracted and used for training and classification using the machine learning classifier `SVM`.
- The overall flow diagram of the proposed method is shown in the following figure. ![flow diagram](https://github.com/SohamChattopadhyayEE/Breast-cancer-detection-using-ResNet_SVM/blob/main/Pictures/Slide1.JPG)

## Dependencies
    pip install -r requirements.txt

## Folder structure
    Root folder
        |-----> static                          # The static folder containing unchangable frontend codes
        |          |-----> css/index.css        # CSS file for frontend
        |          |-----> downloads            # contains .png, .jpg, .jpeg files to be downloaded
        |          |-----> js/index.js          # JavaScript file for forntend
        |          |-----> uploads              # contains .png, .jpg, .jpeg files to be uploaded
        |-----> templates/index.html            # the HTML file
        |-----> utils                           # Contains .py files and weights of trained ResNet18 
        |         |                              (for features) and trained Support Vector Machine (SVM for classification)  
        |         |-----> classification        # Contains the svm classifier model for training and prediction
        |         |        |-> train.py         # Python code for training of SVM
        |         |        |-> svm_clf.py       # Python code for prediction of cancer
        |         |-----> ResNet18_features.py  # Python code for features extraction from ResNet18
        |         |-----> model_ResNet18.pt     # Trained weights of ResNet18
        |         |-----> svm_clf.pkl           # Trained and saved SVM model
        |         |-----> svm_clf_train_val.pkl # Model saved during validation (can be ignored)
        |-----> detector.py                     # Compunded Python file to perform the prediction 
        |-----> app.py                          # The flask app to integrate frontend with backend


