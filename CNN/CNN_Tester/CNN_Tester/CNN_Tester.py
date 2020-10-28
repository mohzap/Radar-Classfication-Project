import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.optim as optim
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from numpy import save
import sys
import h5py
from pathlib import Path

#Location of model file
MODEL_PATH = 'C:\\Users\\Mohamed Zahier\\Documents\\UCT\\4th year\\EEE4022S\\Thesis\\Code\\GitHub\\CNN\\CNN_Tester\\Model\conv_net_model.ckpt'



#Have to give learning rate for criterion
learning_rate = 0.0005

#Loading training, valadation and testing data
#Load train data that is more then 2Gb and hence uses hpf5 format(Matlab save uses -V7.3 hence hpf5 compression)
train_dataset = h5py.File('train_data.mat','r')
ref=train_dataset['train_data'][0][:]
temp_train_dataset=temp_train_dataset=np.empty((len(ref), train_dataset[ref[0]].shape[1], train_dataset[ref[0]].shape[0]))
for i in range(len(ref)):
    temp_train_dataset[i]=np.transpose(np.array(train_dataset[ref[i]]))
train_dataset.close()
x_train=temp_train_dataset;

val_dataset = loadmat('val_data.mat')
val_dataset=val_dataset['val_data']
val_dataset=np.squeeze(val_dataset)
temp_val_dataset=np.empty((val_dataset.shape[0], val_dataset[0].shape[0], val_dataset[0].shape[1]))
for i in range(0,len(val_dataset)):
    temp_val_dataset[i]=val_dataset[i]
x_val=temp_val_dataset;

test_dataset = loadmat('test_data.mat')
test_dataset=test_dataset['test_data']
test_dataset=np.squeeze(test_dataset)
temp_test_dataset=np.empty((test_dataset.shape[0], test_dataset[0].shape[0], test_dataset[0].shape[1]))
for i in range(0,len(test_dataset)):
    temp_test_dataset[i]=test_dataset[i]
x_test=temp_test_dataset;

#Loading training, valadation and testing labels
train_labels = loadmat('train_labels.mat')
train_labels=train_labels['train_labels']
train_labels=np.squeeze(train_labels)
for i in range(0,len(train_labels)):
    train_labels[i]=train_labels[i][0]
y_train=train_labels;
val_labels = loadmat('val_labels.mat')
val_labels=val_labels['val_labels']
val_labels=np.squeeze(val_labels)
for i in range(0,len(val_labels)):
    val_labels[i]=val_labels[i][0]
y_val=val_labels;
test_labels = loadmat('test_labels.mat')
test_labels=test_labels['test_labels']
test_labels=np.squeeze(test_labels)
for i in range(0,len(test_labels)):
    test_labels[i]=test_labels[i][0]
y_test=test_labels;
#Determine Class Weigths
classes,classescount=np.unique(y_train,return_counts=True)
class_weights= torch.empty(len(classes))
for i in range(0,len(classes)):
    cw=max(classescount)/classescount[i]
    class_weights[i]=cw



#Custom Dataset
class MSTAR(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.transform = transform
        self.enc = LabelEncoder()
        targets = self.enc.fit_transform(targets.reshape(-1,))
        self.targets = torch.LongTensor(targets)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = int(index.item())
            
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index])
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

#Dataloader
transform = transforms.Compose(
    [
     transforms.ToTensor()
    ])

# define datasets
train_dataset = MSTAR(x_train, y_train, transform=transform)
val_dataset = MSTAR(x_val, y_val, transform=transform)
test_dataset = MSTAR(x_test, y_test, transform=transform)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=150, shuffle=False)
val_loader = DataLoader(dataset=val_dataset, batch_size=150, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=150, shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

model = ConvNet()
#Load model
#model.load_state_dict(torch.load(MODEL_PATH),strict=False)
#model.eval()
state_dict=torch.load(MODEL_PATH)
#print(state_dict.keys())
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():  #Fix incorrect labeling during saving
    #print(k)
    #print(k.find('fc')!=-1)
    if(k.count('.')==2 and k.find('fc')!=-1):
        loc=k.find('.')
        name=k[0:loc]+k[loc+2:]
    else:
        name=k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval()
#Move to GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
model=model.to(device)

# Loss and optimizer
class_weights=class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Evalute accuracy and loss function
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
    # Return mean loss, accuracy
    return running_loss / len(loader), correct / total




#Train values final 
trai_loss, trai_acc = evaluate(model, train_loader)
print('Final Training Loss: {}, Final Training Accuracy of the model: {} %'.format(trai_loss,trai_acc* 100))

#Validation values final 
vali_loss, vali_acc = evaluate(model, val_loader)
print('Final Validation Loss: {}, Final Validation Accuracy of the model: {} %'.format(vali_loss,vali_acc* 100))

#Test model values final 
test_loss, test_acc = evaluate(model, test_loader)
print('Test Loss: {}, Test Accuracy of the model: {} %'.format(test_loss,test_acc* 100))

#Confusion matrix plot function
def Confusion_Matrix(model, loader, dataset, name):
    model.eval()
    y_tot=torch.empty(0)
    y_pred_tot=torch.empty(0)
    with torch.no_grad():
        for data in loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs=model(x)
            outputs=outputs.cpu()
            _, y_pred = torch.max(outputs, 1)
            y=y.cpu()
            y_tot = torch.cat((y_tot, y), 0)
            y_pred_tot = torch.cat((y_pred_tot, y_pred), 0)

    cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy())
    np.set_printoptions(precision=4)
    print(cm)

    # Coloured confusion matrix
    plt.figure(figsize = (12,12))
    cm = confusion_matrix(y_tot.numpy(), y_pred_tot.numpy(), normalize="true")
    plt.matshow(cm, fignum=1)

    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    
    plt.xticks(range(6))
    plt.yticks(range(6))
    plt.xlabel("Prediction")
    plt.ylabel("True")

    # We can retrieve the categories used by the LabelEncoder
    classes = dataset.enc.classes_.tolist()
    plt.gca().set_xticklabels(classes)
    plt.gca().set_yticklabels(classes)

    plt.title("Normalized Confusion Matrix For "+name+" Data")
    plt.colorbar()
    plt.show()



# Train Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
Confusion_Matrix(model,train_loader,train_dataset,"Train")




# Validation Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
Confusion_Matrix(model,val_loader,val_dataset,"Validation")
  

# Test Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
Confusion_Matrix(model,test_loader,test_dataset,"Test")
   
