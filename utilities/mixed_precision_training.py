import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_chanels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=420,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(
            in_channels=420,
            out_channels=1000,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1)
        )
        self.fc1 = nn.Linear(1000* 7 * 7, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

#Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Hyperparameter
in_channel = 1
num_classes = 10
learning_rate = 3e-4
batch_size = 10
num_epochs = 5

#Load data
train_dataset = datasets.MNIST(
    root="/mnt/c/Users/Prant/workspace/dataset", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root="/mnt/c/Users/Prant/workspace/dataset", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Initalize the model
model = CNN().to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Necessary for FP16
sacler = torch.cuda.amp.GradScaler()

# Train Model
for epoch in range(num_epochs):
    for data, targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = criterion(scores, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)

    print(
        f" Got {num_correct} / {num_samples} with accuracy {(float(num_correct)/float(num_samples))*100:.2f}"
    )

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)