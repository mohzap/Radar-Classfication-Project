
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

# Hyperparameters
num_epochs = 20
num_classes = 4
batch_size = 10
learning_rate = 0.001

#Model save location
MODEL_STORE_PATH = 'C:\\Users\\Mohamed Zahier\\Documents\\UCT\\4th year\\EEE4022S\\Thesis\\Code\\Attempt1\\Model\\'

# Prepare dataset and labels
#Load data from Matlab file(.mat)
dataset = loadmat('data.mat')
dataset=dataset['data']
dataset=np.squeeze(dataset)
#dataset=np.empty((dataset.shape[0], dataset[0].shape[0], dataset[0].shape[1]))
x_train=dataset;
labels = loadmat('labels.mat')
labels=labels['labels']
labels=np.squeeze(labels)
#Remove array enclosing individual labels
for i in range(0,len(labels)):
    labels[i]=labels[i][0]
y_train=labels;

#Splitting dataset
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

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

#Data scaling
'''
scaler = MinMaxScaler()
x_train_shape = x_train.shape
x_val_shape = x_val.shape
x_test_shape = x_test.shape

x_train = scaler.fit_transform(x_train.reshape(-1, 1))
x_train = x_train.reshape(x_train_shape)

x_val = scaler.transform(x_val.reshape(-1, 1))
x_val = x_val.reshape(x_val_shape)

x_test = scaler.transform(x_test.reshape(-1, 1))
x_test = x_test.reshape(x_test_shape)
'''


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
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Sequential(
            nn.Linear(3 * 5 * 64, 256),
            nn.ReLU())
        self.fc2 = nn.Linear(256, 4)

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
#Move to GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
model=model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
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
# Train the model
total_step = len(train_loader)
history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
}
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #history['train_loss'].append(loss.item())
        trainl=loss.item(); #When appending for logging iterations
        

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the training accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        #history['train_acc'].append(correct / total)
        trainacc=correct / total

        #Log data after every 10 iterations
        if (i + 1) % 10 == 0:
            history['train_loss'].append(trainl)
            history['train_acc'].append(trainacc)
            # Determine validation accuracy and loss
            val_loss, val_acc = evaluate(model, val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Train Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, trainl,
                          trainacc * 100))
            print('                           Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%'
                  .format( val_loss,
                          val_acc*100))
# Test the model (!!NB Redundant to evaluate fucntion therefore replace with evaluate)
'''
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format((correct / total) * 100))
'''
#Test model alternative
test_loss, test_acc = evaluate(model, test_loader)
print('Test Accuracy of the model on the 10000 test images: {} %'.format(test_acc* 100))


# Save the model
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

#Save test data and labels
save('test_data.npy',x_test)
save('test_labels.npy',y_test)
# !!!Plot results!!!

# Loss graph
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel("Logging iterations")
plt.ylabel("Cross-entropy Loss")
plt.legend()
plt.show()

# Accuracy graph
fig = plt.figure(figsize=(8,8))
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.xlabel("Logging iterations")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Make device CPU again for confusion matrix
device="cpu"
model=model.to(device)
# Confusion matrix
# In this case we know there will only be one batch consisting of the entire test set
it = iter(test_loader)
x, y = next(it)
x, y = x.to(device), y.to(device)
outputs = model(x)
_, y_pred = torch.max(outputs, 1)

cm = confusion_matrix(y.numpy(), y_pred.numpy())
np.set_printoptions(precision=4)
print(cm)

# Coloured confusion matrix
plt.figure(figsize = (10,10))
cm = confusion_matrix(y.numpy(), y_pred.numpy(), normalize="true")
plt.matshow(cm, fignum=1)

for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    
plt.xticks(range(4))
plt.yticks(range(4))
plt.xlabel("Prediction")
plt.ylabel("True")

# We can retrieve the categories used by the LabelEncoder
classes = test_dataset.enc.classes_.tolist()
plt.gca().set_xticklabels(classes)
plt.gca().set_yticklabels(classes)

plt.title("Normalized Confusion Matrix")
plt.colorbar()
plt.show()
