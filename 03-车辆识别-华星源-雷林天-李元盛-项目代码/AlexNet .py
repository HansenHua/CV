import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset3 import VeriDataset


image_size = 100
input_size = 3 * image_size * image_size
training_iteration = 10
batch_size = 32 
LR = 1e-5

ImagePath_veri="../VehicleID/image/"
TrainList_veri="../VehicleID/train_test_split/3train16000.txt"
TestList_veri= "../VehicleID/train_test_split/3test4000.txt"

device = device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()

        # self.feature = nn.Sequential(

        #     nn.Conv2d(3, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(6, 18, kernel_size=5), nn.Sigmoid(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.classifier=nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(19044, 512), nn.Sigmoid(),
        #     nn.Linear(512, 256), nn.Sigmoid(),
        #     nn.Linear(256, 2),
        #     nn.Softmax(dim=1)
        # )

        self.feature = nn.Sequential(
             # 这里，我们使用一个11*11的更大窗口来捕捉对象。
            # 同时，步幅为4，以减少输出的高度和宽度。
            # 另外，输出通道的数目远大于LeNet
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 使用三个连续的卷积层和较小的卷积窗口。
            # 除了最后的卷积层，输出通道的数量进一步增加。
            # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
            nn.Linear(2048, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Dropout(p=0.5),
            # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(2048, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, image1, image2):
        
        feature1 = self.feature(image1)
        feature2 = self.feature(image2)
        feature = torch.concat([feature1, feature2],dim=1)
        output = self.classifier(feature)

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
        CNN_net = CNN_Net().to(device).eval()
        optimizer_AE = torch.optim.Adam(CNN_net.parameters(), lr = LR)
        # 训练分类器
        for _ in range(training_iteration):
            dataloader = get_dataset(ImagePath_veri, TrainList_veri)
            for image1,image2, label in dataloader:

                image1 = image1.to(device)
                image2 = image2.to(device)
                
                image1 = image1.reshape(-1,3,image_size,image_size)
                image2 = image2.reshape(-1,3,image_size,image_size)
                label = label.to(device)
                pre_label =  CNN_net(image1, image2)
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
                image1 = image1.to(device)
                image2 = image2.to(device)
                label = label.to(device)
                image1 = image1.reshape(-1,3,image_size,image_size)
                image2 = image2.reshape(-1,3,image_size,image_size)
                pre_label =  CNN_net(image1, image2)
                
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pre_label, label)
                acc+=torch.sum(torch.argmax(pre_label,dim=1)==label).item()
                sum+=len(label)
                loss_sum+=loss.item()
            if acc > bestacc:
                bestacc = acc
                torch.save(CNN_net, 'BEST-CNN.pth')
            print('test acc: %.2f%%, loss: %.4f'%(100*acc/sum, loss_sum/(batch+1)))

        
    else:
        pass

if __name__ == "__main__":
    # Path to Images
    print("start training CNN")
    print("training_iteration="+str(training_iteration))
    print("batch_size=" + str(batch_size))
    print("image_size=" + str(image_size))
    print()
    main('train')