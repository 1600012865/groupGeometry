import torch
import torchvision.datasets as dsets
from torchvision import transforms
import torch.nn as nn
import os

def add_noise(img,std=1,p=0.1):
    img_size = img.shape
    mask = np.random.uniform(0,10,size=img_size)
    mask = mask < p
    img_noise = std * np.random.randn(*img_size)
    img_noise = img_noise * mask
    nimg = img + img_noise
    nimg = np.clip(nimg,0,255)
    return nimg

def cal_grad(net):
    res = 0
    for p in net.parameters():
        res += (p.grad.view(-1) ** 2).sum()
    return res ** (1/2)

class mydataset(nn.Module):

  def __init__(self,img_path, max_degrade_num, max_p, min_p, max_std, min_std):
    self.img_path = img_path
    self.file_name = os.listdir(img_path)
    self.transform = transform
    self.epoch = -1
    self.cur_std = 0
    self.cur_p = 0
    self.std_list = np.linspace(0,np.pi/2,max_degrade_num)
    self.p_list = np.linspace(0,np.pi/2,max_degrade_num)
    self.std_list = max_std + (max_std - min_std) *(np.cos(self.std_list)-1)
    self.p_list = max_p + (max_p - min_p) * (np.cos(self.p_list)-1)



  def __len__(self):
    return len(self.file_name)

  def __getitem__(self,idx):

    name = os.path.join(self.img_path,self.file_name[idx])
    img = cv2.imread(name,0)
    img = add_noise(img,self.cur_std,self.cur_p)
    img = (img/255 - 0.5)*2
    
    return torch.tensor(img).float().unsqueeze(0)

    
