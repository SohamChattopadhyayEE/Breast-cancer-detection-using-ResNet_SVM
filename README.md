# Automatic Breast Cancer Detection From Hystopathology Images
This is a Python project for breast cancer detection. 

![gif](https://github.com/SohamChattopadhyayEE/Breast-cancer-detection-using-ResNet_SVM/blob/main/videos/Benign.gif)

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


