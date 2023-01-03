import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VeriDataset
from dataset3 import VeriDataset as VeriDataset3

class_number = 28000
image_size = 100
training_iteration = 10
Layers = [3, 4, 6, 3]
batch_size=128

ImagePath_veri="../VehicleID/image/"
TrainList_veri="../VehicleID/train_test_split/train_list.txt"
TestList_veri= "../VehicleID/train_test_split/3test.txt"

class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        filter1, filter2, filter3 = filters
        self.is_1x1conv = is_1x1conv
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(filter3),
        )
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride,  bias=False),
                nn.BatchNorm2d(filter3)
            )
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)
        x = x + x_shortcut
        x = self.relu(x)
        return x


class Resnet50(nn.Module):

    def __init__(self):
        super(Resnet50,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = self._make_layer(64, (64, 64, 256), Layers[0])
        self.conv3 = self._make_layer(256, (128, 128, 512), Layers[1], 2)
        self.conv4 = self._make_layer(512, (256, 256, 1024), Layers[2], 2)
        self.conv5 = self._make_layer(1024, (512, 512, 2048), Layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, class_number)
        self.my_relu = nn.ReLU()
        self.my_softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.my_relu(self.fc1(x))
        x = self.my_relu(self.fc2(x))
        x = self.fc3(x)
        output = self.my_softmax(x)
        return output
    
    def _make_layer(self, in_channels, filters, blocks, stride=1):
        layers = []
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)
        for _ in range(1, blocks):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))

        return nn.Sequential(*layers)

def get_dataset(data_dir,train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    train_set = VeriDataset(data_dir, train_list, train_data_transform, is_train= True )

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    
    return train_loader

def get_dataset3(data_dir,train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    train_set = VeriDataset3(data_dir, train_list, train_data_transform, is_train= True )

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, )
    
    return train_loader



print("start training ResNet")
print("training_iteration=" + str(training_iteration))
print("batch_size=" + str(batch_size))
print("image_size=" + str(image_size))
print()


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = Resnet50().to(device).eval()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
loss_func = nn.CrossEntropyLoss()


for _ in range(training_iteration):
    #train
    dataloader = get_dataset(ImagePath_veri, TrainList_veri)
    for batch,(sample_image, sample_label) in enumerate(dataloader):
        sample_image = sample_image.to(device)
        sample_label = sample_label.to(device)
        batchsize = len(sample_label)
        sample_image = sample_image.reshape(-1,3,image_size,image_size)
        pre = model(sample_image)
        loss = loss_func(pre, sample_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch == 100:
            break
    
    #test
    acc = 0.0
    sum = 0.0
    bestacc = 0
    criteria = nn.MSELoss()
    testloader = get_dataset3(ImagePath_veri, TestList_veri)
    for batch,(image1,image2, label) in enumerate(testloader):
        batchsize = len(label)
        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)
        image1 = image1.reshape(-1,3,image_size,image_size)
        image2 = image2.reshape(-1,3,image_size,image_size)
        pre_label1 =  model(image1)
        pre_label2 =  model(image2)
        pre = (torch.argmax(pre_label1,dim=1) == torch.argmax(pre_label2,dim=1))
        acc+=torch.sum(pre==label).item()
        sum+=len(label)
        
    if acc > bestacc:
        bestacc = acc
        torch.save(model, 'BEST-ResNet.pth')
    print('test acc: %.2f%%'%(100*acc/sum))

    
