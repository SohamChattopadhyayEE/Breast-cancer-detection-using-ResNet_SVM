import cv2
import os
import torch
import torchvision
import torch.nn as nn

import utils.classification.svm_clf as clf

load_model = 'utils/model_ResNet18.pt'

models = torchvision.models.resnet18(pretrained=True)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        img_modules = list(models.children())[:-1]
        self.ModelA = nn.Sequential(*img_modules)
        self.relu = nn.ReLU()
        self.Linear3 = nn.Linear(512, 2, bias = True)

    def forward(self, x):
        x = self.ModelA(x) # N x 1024 x 1 x 1
        x1 = torch.flatten(x, 1) 
        x2 = self.Linear3(x1)
        return  x1, x2

net = MyModel()
params = net.parameters()
optimizer=torch.optim.Adam(net.parameters())

if os.path.exists(load_model):
    checkpoint=torch.load(load_model,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True

def get_features(img):
    net = MyModel()
    img_arr = cv2.imread(img)
    img_arr = cv2.resize(img_arr, (224,224))
    img_arr = torch.tensor(img_arr)
    img_arr = img_arr.permute(2,0,1)
    img = img_arr.unsqueeze(0)
    net = net.cpu()
    X,_ = net(img/255)
    X = X.cpu().detach()
    return X

if __name__=="__main__":
    path = 'Sample_data/SOB_B_A-14-22549AB-400-005.png'
    feature = get_features(path)
    print(feature.shape)
    pred = clf.predict(feature)
    print(pred)