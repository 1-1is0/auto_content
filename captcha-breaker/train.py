# %%
import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision import models
import torch.nn as nn
from fastai.vision.all import Path
import torch
from torch import Tensor

device = torch.device("cuda:0" torch.cuda.is_available() else "cpu")
print(device)
# %%

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 5


def encode(a):
    onehot = [0]*ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] += 1
    return onehot

class Mydataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.img = os.listdir(self.path)
        self.transform = transform
        
    def __getitem__(self, idx):
        img_path = self.img[idx]
        img = Image.open(self.path/img_path)
        img = img.convert('L')
        label = Path(self.path/img_path).name[:-4]
        label_oh = []
        for i in label:
            label_oh += encode(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label
    
    def __len__(self):
        return len(self.img)

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

train_ds = Mydataset(Path('./data/train'), transform=transform)
test_ds = Mydataset(Path('./data/test'), transform=transform)
train_dl = DataLoader(train_ds, batch_size=64, num_workers=0)
test_dl = DataLoader(train_ds, batch_size=1, num_workers=0)

# %%

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN*MAX_CAPTCHA, bias=True)

# %%

PATH="/kaggle/working/weights.txt"
torch.save(model.state_dict(), PATH)
model.load_state_dict(torch.load(PATH))
model.eval()


# %%
model = model.to(device)

loss_func = nn.MultiLabelSoftMarginLoss()
optm = torch.optim.Adam(model.parameters(), lr=0.001)


# %%
for epoch in range(20):
    for step, i in enumerate(train_dl):
        img, label_oh, label = i
        img = Tensor(img).to(device)
        label_oh = Tensor(label_oh.float()).to(device)
        pred = model(img)
        loss = loss_func(pred, label_oh)
        optm.zero_grad()
        loss.backward()
        optm.step()
        print('eopch:', epoch+1, 'step:', step+1, 'loss:', loss.item())

# %%
import torch
import gc
# del variables
gc.collect()
torch.cuda.empty_cache()


# %%
for step, (img, label_oh, label) in enumerate(test_dl):
    img = Tensor(img).cuda()
    pred = model(img)

    c0 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[0:ALL_CHAR_SET_LEN])]
    c1 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN:ALL_CHAR_SET_LEN*2])]
    c2 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*2:ALL_CHAR_SET_LEN*3])]
    c3 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*3:ALL_CHAR_SET_LEN*4])]
    c4 = ALL_CHAR_SET[np.argmax(pred.squeeze().cpu().tolist()[ALL_CHAR_SET_LEN*4:ALL_CHAR_SET_LEN*5])]
    c = '%s%s%s%s%s' % (c0, c1, c2, c3, c4)

    print('label:', label[0], 'pred:', c)