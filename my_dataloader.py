import torch.utils.data as data
import os
import os.path
import torch
import numpy as np


class MyDataset(data.Dataset):
    def __init__(self, root, npoints=2048):
        self.npoints = npoints
        self.root = root
        self.base_filepaht = 'crop/2048/'
        self.datapathc2048 = []
        self.datapathc1024 = []
        self.datapathc512  = []
        self.datapathr512  = []
        self.datapathr128  = []
        self.datapathr64   = []
        filelist = os.listdir(self.root+self.base_filepaht)
        for filename in filelist:
            filepath = os.path.abspath(os.path.join(self.root,self.base_filepaht, filename))
            self.datapathc2048.append(filepath)
            filepath = os.path.abspath(os.path.join(self.root, "crop","1024", filename))
            self.datapathc1024.append(filepath)
            filepath = os.path.abspath(os.path.join(self.root, "crop","512", filename))
            self.datapathc512.append(filepath)
            filepath = os.path.abspath(os.path.join(self.root, "real","512", filename))
            self.datapathr512.append(filepath)
            filepath = os.path.abspath(os.path.join(self.root, "real","128", filename))
            self.datapathr128.append(filepath)
            filepath = os.path.abspath(os.path.join(self.root, "real","64", filename))
            self.datapathr64.append(filepath)
        
    def __getitem__(self, index):
        fnc2048 = self.datapathc2048[index]
        fnc1024 = self.datapathc1024[index]
        fnc512  = self.datapathc512[index]
        fnr512  = self.datapathr512[index]
        fnr128  = self.datapathr128[index]
        fnr64   = self.datapathr64[index]
        point_c2048 = np.loadtxt(fnc2048).astype(np.float32)
        point_c2048 = torch.from_numpy(point_c2048)
        point_c1024 = np.loadtxt(fnc1024).astype(np.float32)
        point_c1024 = torch.from_numpy(point_c1024)
        point_c512  = np.loadtxt(fnc512).astype(np.float32)
        point_c512  = torch.from_numpy(point_c512)
        point_r512  = np.loadtxt(fnr512).astype(np.float32)
        point_r512  = torch.from_numpy(point_r512)
        point_r128  = np.loadtxt(fnr128).astype(np.float32)
        point_r128  = torch.from_numpy(point_r128)
        point_r64   = np.loadtxt(fnr64).astype(np.float32)
        point_r64   = torch.from_numpy(point_r64)
        return point_c2048,point_c1024,point_c512,point_r512,point_r128,point_r64
    
    def __len__(self):
        return len(self.datapathc2048)

if __name__ == '__main__':
    dset = MyDataset( root='./dataset/resample/train_1/', npoints=2048)
    print(dset)