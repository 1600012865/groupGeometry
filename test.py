import numpy as np
import torch
from sagan_models import Generator, Discriminator
import os
import torch.nn as nn
from metric import get_metric

b_size = 1000
z = 32
path = '/data-174/xuanjc/geometry/models/sagan_rec0'

def getsecond(ele):
  return ele[1]
  
G = Generator(z_dim=z)
G.cuda()
G = nn.DataParallel(G)
models = os.listdir(path)
modelsG = [i for i in models if 'G' in i]
modelsG.sort()
length = len(modelsG)
res_list = []
for i in range(length):
  try:
    G.load_state_dict(torch.load(os.path.join(path,modelsG[i])))
  except:
    print('degraded: %s'%(modelsG[i]))
    continue
  else:
    pass
  noise = torch.randn(b_size, z, 1, 1)
  noise = noise.cuda()
  fake = G(noise)
  fake = fake/2 + 0.5
  fake = fake.cpu().detach().numpy()
  res = 0
  for j in range(b_size):
    res += get_metric(fake[j,0])[0]
  res_list.append((modelsG[i],res/b_size))
  print('%d/%d: %.3f'%(i+1,length,res/b_size))
res_list.sort(key=getsecond)
print(res_list)
torch.save(res_list,'metric_result.pkl')
print(res)
