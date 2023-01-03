import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from dataset3 import VeriDataset
from torchvision import utils as vutils

# Hyperparameters
image_size = 100
input_size = 3 * image_size * image_size
training_iteration = 10
batch_size = 32 
LR = 1e-5

ImagePath_veri="../VehicleID/image/"
TrainList_veri="../VehicleID/train_test_split/3train16000.txt"
TestList_veri= "../VehicleID/train_test_split/3test4000.txt"

device = device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.my_relu = nn.ReLU()
        self.my_flattern = nn.Flatten()
    
    def forward(self, image):
        x = self.my_flattern(image)
        x = self.fc1(x)
        x = self.my_relu(x)
        x = self.fc2(x)
        x = self.my_relu(x)
        output = self.fc3(x)

        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, input_size)
        self.my_relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.my_relu(x)
        x = self.fc2(x)
        x = self.my_relu(x)
        output = self.fc3(x)

        return output

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image):
        feature = self.encoder(image)
        img = self.decoder(feature)

        return img

    def extract_feature(self, image):
        feature = self.encoder(image)
        
        return feature

    def cal_loss(self, image):
        feature = self.encoder(image)
        img = self.decoder(feature)

        loss = F.mse_loss(image, img)
        return loss

class AE_Net(nn.Module):
    def __init__(self, encoder):
        super(AE_Net, self).__init__()
        self.encoder = encoder
        # TODO:分类器
        self.classify = nn.Sequential(
            nn.Linear(2*512, 512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(2*512, 512)
        self.fc2 = nn.Linear(512,64)
        self.fc3 = nn.Linear(64, 2)
        self.my_relu = nn.ReLU()
        self.my_softmax = nn.Softmax(dim=1)

    def forward(self, image1, image2):
        feature1 = self.encoder.extract_feature(image1)
        feature2 = self.encoder.extract_feature(image2)
        feature = torch.concat([feature1, feature2],dim=1)
        output = self.classify(feature)

        return output

def get_dataset(data_dir,train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_data_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    train_set = VeriDataset(data_dir, train_list, train_data_transform, is_train= True )

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, )
    
    return train_loader


    

def main(mode):
    if mode == 'train':
        encoder = AutoEncoder().to(device).eval()
        optimizer_encoder = torch.optim.Adam(encoder.parameters(), lr = LR)
        # 先训练编码器
        for _ in range(training_iteration):
            # 取样
            dataloader = get_dataset(ImagePath_veri, TrainList_veri)
            for image1,image2, label in dataloader:
                # 训练encoder
                # print(image1.shape,image2.shape,label)
                batchsize = len(label)
                image1 = image1.to(device)
                image2 = image2.to(device)
                image1 = image1.reshape(batchsize,3*image_size*image_size)
                image2 = image2.reshape(batchsize,3*image_size*image_size)

                loss = encoder.cal_loss(image1)
                optimizer_encoder.zero_grad()
                loss.backward()
                optimizer_encoder.step()

                loss = encoder.cal_loss(image2)
                optimizer_encoder.zero_grad()
                loss.backward()
                optimizer_encoder.step()
            print("AUTOENCODER"+str(_))
                
        
        #展示Auto效果
        # def save_image_tensor(input_tensor: torch.Tensor, filename):
        #     """
        #     将tensor保存为图片
        #     :param input_tensor: 要保存的tensor
        #     :param filename: 保存的文件名
        #     """
        #     assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
        #     # 复制一份
        #     input_tensor = input_tensor.clone().detach()
        #     # 到cpu
        #     input_tensor = input_tensor.to(torch.device('cpu'))
        #     # 反归一化
        #     # input_tensor = unnormalize(input_tensor)
        #     vutils.save_image(input_tensor, filename)

        # img1 = Image.open('../VehicleID/image/0104211.jpg').convert('RGB')
        # img1 = train_data_transform(img1).reshape(1,3,image_size,image_size)
        # save_image_tensor(img1,"./sor.jpg")
        # cpu_encoder = encoder.to("cpu")
        # img1 = img1.reshape(1,3*image_size*image_size)
        # out = cpu_encoder(img1).detach().reshape(1,3,image_size,image_size)
        # save_image_tensor(out,"./out.jpg") 

        
        AE_net = AE_Net(encoder).to(device).eval()
        optimizer_AE = torch.optim.Adam(AE_net.parameters(), lr = LR)
        # 训练分类器
        for _ in range(training_iteration):
            dataloader = get_dataset(ImagePath_veri, TrainList_veri)
            for image1,image2, label in dataloader:
                batchsize = len(label)
                image1 = image1.to(device)
                image2 = image2.to(device)
                
                image1 = image1.reshape(batchsize,3*image_size*image_size)
                image2 = image2.reshape(batchsize,3*image_size*image_size)
                label = label.to(device)
                pre_label =  AE_net(image1, image2)
                

                criteria = nn.CrossEntropyLoss()
                loss = criteria(pre_label, label)

                optimizer_AE.zero_grad()
                loss.backward()
                optimizer_AE.step()
            
            acc = 0.0
            sum = 0.0
            loss_sum = 0
            bestacc = 0
            testloader = get_dataset(ImagePath_veri, TestList_veri)
            for batch,(image1,image2, label) in enumerate(testloader):
                batchsize = len(label)
                image1 = image1.to(device)
                image2 = image2.to(device)
                label = label.to(device)
                image1 = image1.reshape(batchsize,3*image_size*image_size)
                image2 = image2.reshape(batchsize,3*image_size*image_size)
                pre_label =  AE_net(image1, image2)
                
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pre_label, label)
                acc+=torch.sum(torch.argmax(pre_label,dim=1)==label).item()
                sum+=len(label)
                loss_sum+=loss.item()
            if acc > bestacc:
                bestacc = acc
                torch.save(AE_Net, 'BEST-AUTO.pth')
            print('test acc: %.2f%%, loss: %.4f'%(100*acc/sum, loss_sum/(batch+1)))

        
    else:
        pass

if __name__ == "__main__":
    # Path to Images
    print("start training Autoencoder")
    print("training_iteration="+str(training_iteration))
    print("batch_size=" + str(batch_size))
    print("image_size=" + str(image_size))
    print()
    main('train')