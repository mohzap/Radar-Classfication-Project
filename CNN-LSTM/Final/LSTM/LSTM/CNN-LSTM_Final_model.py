
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
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from numpy import save
import sys
import h5py

# Hyperparameters
num_epochs = 40
num_classes = 6
batch_size = 75
learning_rate = 0.0001  
timesteps=128
# Input Size 128*128

#Model save location
MODEL_STORE_PATH = 'C:\\Users\\Mohamed Zahier\\Documents\\UCT\\4th year\\EEE4022S\\Thesis\\Code\\LSTM\\Model\\'


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
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# LSTM1 model
class LSTM1(nn.Module):
    def __init__(self):
        super(LSTM1, self).__init__()
        self.output_size = 6
        self.n_layers = 2
        self.hidden_dim = 128
        self.layer1 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer6 = nn.LSTM(input_size=128, hidden_size=self.hidden_dim, num_layers=self.n_layers ,batch_first=True)
        self.drop_out = nn.Dropout()
        self.fc = nn.Linear(128, 6)
        self.hidden_cell = (torch.zeros(self.n_layers, batch_size,self.hidden_dim),
                            torch.zeros(self.n_layers, batch_size,self.hidden_dim))

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size() #H is frequancy, W is time
        c_in=x.view(batch_size * timesteps, C, H, W)
        c_in=torch.squeeze(c_in.permute(0,2, 1, 3)) #Swop dimensions(C is frequancy and W is time)
        c_out1 = self.layer1(c_in)
        c_out2 = self.layer2(c_out1)
        c_out3 = self.layer3(c_out2)
        c_out4 = self.layer4(c_out3)
        c_out5 = self.layer5(c_out4)
        r_in= c_out5.view(batch_size, -1, H)
        r_out,self.hidden_cell = self.layer6(r_in,self.hidden_cell)
        r_out=r_out[:, -1, :] #Get last hidden state from LSTM layer
        r_out = self.drop_out(r_out)
        out = self.fc(r_out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden



model = LSTM1()

#Determine the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
#Move to device GPU
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
            inputs = np.expand_dims(inputs, axis=1)
            inputs = torch.FloatTensor(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            model.hidden_cell = (torch.zeros(model.n_layers, inputs.size(0), model.hidden_dim).to(device),
                            torch.zeros(model.n_layers, inputs.size(0), model.hidden_dim).to(device))        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
        
    # Return mean loss, accuracy
    return running_loss / len(loader), correct / total
# Train the model
total_step = len(train_loader)
history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
}
for epoch in range(num_epochs):
    train_acc_av=[]
    train_loss_av=[]
    val_acc_av=[]
    val_loss_av=[]
    for i, (images, labels) in enumerate(train_loader):
        model.hidden_cell = (torch.zeros(model.n_layers, images.size(0), model.hidden_dim).to(device),
                        torch.zeros(model.n_layers, images.size(0), model.hidden_dim).to(device)) 
        images = np.expand_dims(images, axis=1) #Place in channel variable 
        images = torch.FloatTensor(images)
        images, labels = images.to(device), labels.to(device)
        model.train()
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        trainl=loss.item(); #When appending for logging iterations
        

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the training accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        trainacc=correct / total
        train_loss_av.append(trainl)
        train_acc_av.append(trainacc)

        #Log data after every 10 iterations
        if (i + 1) % 10 == 0:
            # Determine validation accuracy and loss
            val_loss, val_acc = evaluate(model, val_loader)
            val_loss_av.append(val_loss)
            val_acc_av.append(val_acc)
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, trainl,
                          trainacc * 100))
            print('                           Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%'
                  .format( val_loss,
                          val_acc*100))
    history['train_loss'].append(sum(train_loss_av)/len(train_loss_av))
    history['train_acc'].append(sum(train_acc_av)/len(train_acc_av))
    history['val_loss'].append(sum(val_loss_av)/len(val_loss_av))
    history['val_acc'].append(sum(val_acc_av)/len(val_acc_av))

#Train values final 
trai_loss, trai_acc = evaluate(model, train_loader)
print('Final Training Loss: {}, Final Training Accuracy of the model: {} %'.format(trai_loss,trai_acc* 100))

#Validation values final 
vali_loss, vali_acc = evaluate(model, val_loader)
print('Final Validation Loss: {}, Final Validation Accuracy of the model: {} %'.format(vali_loss,vali_acc* 100))

#Test model alternative
test_loss, test_acc = evaluate(model, test_loader)
print('Test Loss: {}, Test Accuracy of the model: {} %'.format(test_loss,test_acc* 100))


# Save the model
#torch.save(model.state_dict(), MODEL_STORE_PATH + 'LSTM_model.pt') #Save parameters only 
torch.save(model, MODEL_STORE_PATH + 'LSTM_model.pt') #Save entire model

# !!!Plot results!!!

# Loss graph
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy Loss")
plt.xlim(1,num_epochs)
xi = list(range(num_epochs))
xv = list(range(1,(num_epochs+1)))
plt.xticks(xi, xv)
plt.legend()
plt.show()

# Accuracy graph
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xlim(1,num_epochs)
xi = list(range(num_epochs))
xv = list(range(1,(num_epochs+1)))
plt.xticks(xi, xv)
plt.legend()
plt.show()

#Make device CPU again for confusion matrix
#device="cpu"
#model=model.to(device)
#Confusion matrix plot function
def Confusion_Matrix(model, loader, dataset, name):
    model.eval()
    y_tot=torch.empty(0)
    y_pred_tot=torch.empty(0)
    #x_inc=torch.empty(1,128,129)
    #y_inc=torch.empty(1)
    #y_inc_pred=torch.empty(1)
    #count=0
    #ind=[]
    with torch.no_grad():
        for data in loader:
            x, y = data
            model.hidden_cell = (torch.zeros(model.n_layers, x.size(0), model.hidden_dim).to(device),
                        torch.zeros(model.n_layers, x.size(0), model.hidden_dim).to(device)) 
            x = np.expand_dims(x, axis=1) #Place in channel variable 
            x = torch.FloatTensor(x)
            x, y = x.to(device), y.to(device)
            outputs=model(x)
            outputs=outputs.cpu()
            _, y_pred = torch.max(outputs, 1)
            y=y.cpu()
            y_tot = torch.cat((y_tot, y), 0)
            y_pred_tot = torch.cat((y_pred_tot, y_pred), 0)
            #Work out incorrect predictions
            #for i in range(0,len(y)):
            #    if(y[i]!=y_pred[i]):
            #        x_inc = torch.cat([x_inc,x[i].cpu()], 0)                   
            #        y_inc = torch.cat((y_inc, torch.unsqueeze(y[i],0)), 0)
            #        y_inc_pred = torch.cat((y_inc_pred, torch.unsqueeze(y_pred[i],0)), 0)
            #        ind.append((i+1+count*normal_batch_size))#Index of each element in dataset
            #count=count+1
    #Remove first unused element
    #x_inc=x_inc[1:,:,:]
    #y_inc=y_inc[1:]
    #y_inc_pred=y_inc_pred[1:] 
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
    #return x_inc.numpy(),y_inc.numpy(),y_inc_pred.numpy(),ind

# Train Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
train_loader = DataLoader(dataset=train_dataset, batch_size=150, shuffle=False)# Make new train_dataloader that is not randomised
Confusion_Matrix(model,train_loader,train_dataset,"Train")

'''
# incorrect example determination for Train dataset
x,y_num,y_pred_num,ind=Confusion_Matrix(model,train_loader,train_dataset,"Train")
classes = train_dataset.enc.classes_.tolist()
y=[]
y_pred=[]
for i in range(0,len(y_num)):
    y.append(classes[y_num[i].astype(int)])
    y_pred.append(classes[y_pred_num[i].astype(int)])    
#Save incorrect predictions
train_dict={"x_train_inc":x, "y_train_inc":y, "y_train_inc_pred":y_pred, "train_index":ind}
savemat("incorrect_train.mat",train_dict)

'''

# Validation Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
Confusion_Matrix(model,val_loader,val_dataset,"Validation")

'''
# incorrect example determination for Validation dataset
x,y_num,y_pred_num,ind=Confusion_Matrix(model,val_loader,val_dataset,"Validation")
classes = val_dataset.enc.classes_.tolist()
y=[]
y_pred=[]
for i in range(0,len(y_num)):
    y.append(classes[y_num[i].astype(int)])
    y_pred.append(classes[y_pred_num[i].astype(int)])    
#Save incorrect predictions
val_dict={"x_val_inc":x, "y_val_inc":y, "y_val_inc_pred":y_pred, "val_index":ind}
savemat("incorrect_val.mat",val_dict)
'''

# Test Confusion matrix
# In this case we know there will be multiple batchs consisting of the entire test set
Confusion_Matrix(model,test_loader,test_dataset,"Test")

'''
# incorrect example determination for Test dataset
x,y_num,y_pred_num,ind=Confusion_Matrix(model,test_loader,test_dataset,"Test")
classes = test_dataset.enc.classes_.tolist()
y=[]
y_pred=[]
for i in range(0,len(y_num)):
    y.append(classes[y_num[i].astype(int)])
    y_pred.append(classes[y_pred_num[i].astype(int)])    
#Save incorrect predictions
test_dict={"x_test_inc":x, "y_test_inc":y, "y_test_inc_pred":y_pred, "test_index":ind}
savemat("incorrect_test.mat",test_dict)
'''
