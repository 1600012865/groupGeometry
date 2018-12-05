
from parameter import *
from trainer import Trainer
# from tester import Tester
from dataset import mydataset
import torch
import torch.nn as nn
from torch.backends import cudnn
from utils import make_folder
import os

def main(config):
    # For fast training
    cudnn.benchmark = True


    dataset = mydataset(config.image_path, config.noise_step, config.max_p, config.min_p, config.max_std, config.min_std)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=config.d_batch_size,
                                              shuffle=True,
                                              num_workers=16,
                                              drop_last=True)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)
    make_folder(config.losscurve_path, config.version)
    
    f = open(os.path.join(config.log_path,config.version,'hyperparameters.txt'),'w')
    f.write(str(config))
    f.close()
    
    trainer = Trainer(dataset, data_loader, config)
    trainer.train()
    
if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)