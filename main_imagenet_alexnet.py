import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from alexnet import AlexNet

#hyperparams
batch_size = 128
num_classes = 1000
num_epochs = 10
learning_rate = 0.01 
save_pth = "final_alexnet.pth"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#dataset config
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224), #can also resize + randomcrop
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_valid = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dum_x = torch.randn(10,3,224,224)
dum_y = torch.randint(0,num_classes,(10,))
train_dataset = TensorDataset(dum_x, dum_y)
val_dataset = TensorDataset(dum_x, dum_y)
test_dataset = TensorDataset(dum_x, dum_y)

train_loader = DataLoader(train_dataset,batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size,shuffle=False)
test_loader = DataLoader(test_dataset,batch_size,shuffle=False)

#model
model = AlexNet(in_channels=3, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.0005)
scheduler = ReduceLROnPlateau(optimizer,factor=0.1,patience=2,mode='min') 

#training
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct, total = 0,0

    for i, (images,labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _,predicted = outputs.max(1)
        total += images.size(0)
        correct += (predicted==labels).sum().item()

    return running_loss/total , 100.0 * correct/total #avg loss, accuracy%

def eval(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct, total = 0,0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _,predicted = outputs.max(1)
            total += images.size(0)
            correct += (predicted==labels).sum().item()

    return running_loss/total , 100.0 * correct/total #avg loss, accuracy%

#execution
print(f"Running on device: {device}")
best_val_acc = -1.0 #-1.0 instead of 0.0 bcs dry runs could sometimes result in val_acc = 0.0 throughout and 0.0 > 0.0 will always be false -> model won't save that way

for epoch in range(1,num_epochs+1):
    train_loss,train_acc = train(model,train_loader,criterion,optimizer)
    val_loss,val_acc = eval(model,val_loader,criterion)

    scheduler.step(val_loss)
    curr_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch}/{num_epochs}] | LR: {curr_lr:.6f} | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),save_pth)
        print(f"checkpoint saved at val acc: {val_acc:.2f}")

#testing
if os.path.exists(save_pth):
    model.load_state_dict(torch.load(save_pth))
    print('loaded best model checkpoint for testing')

test_loss, test_acc = eval(model,test_loader,criterion)
print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_acc:.2f}%")






