import os, sys
import torch
import torch.utils.data as data
from PIL import Image

class VeriDataset(data.Dataset):
    def __init__(self, data_dir, train_list, train_data_transform = None, is_train= True ):
        super(VeriDataset, self).__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.train_data_transform = train_data_transform
        reader = open(train_list)
        lines = reader.readlines()
        self.names1 = []
        self.names2 = []
        self.labels = []

        if is_train == True:
            for line in lines:
                line = line.strip().split(' ')
                self.names1.append(line[0])
                self.names2.append(line[1])
                self.labels.append(line[2])
        else:
            pass
        

    def __getitem__(self, index):
        # For normalize

        img1 = Image.open(os.path.join(self.data_dir, self.names1[index])+'.jpg').convert('RGB') # convert gray to rgb
        img2 = Image.open(os.path.join(self.data_dir, self.names2[index])+'.jpg').convert('RGB') # convert gray to rgb
        label = int(self.labels[index])
        

        if self.train_data_transform!=None:
            img1 = self.train_data_transform(img1)
            img2 = self.train_data_transform(img2)
            


        return img1, img2, label

    def __len__(self):
        return len(self.labels)