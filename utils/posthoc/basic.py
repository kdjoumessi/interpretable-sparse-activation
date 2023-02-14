import os
import yaml 
import socket
import numpy as np
import pandas as pd

from PIL import Image
from munch import munchify
from matplotlib import pyplot as plt



import torch
from torch import nn
from bagnets import pytorchnet
from torchvision import models
from torchvision import transforms

from modules.builder import generate_model


#######
#8, 7, 6, 2
hostname = ['2b4a08c67a7e', 'c9a6a71caefe', '32e95f4aafdb', '4905f1483080']

####### Load data, data paths, conf file, and models ################
def data_path(cfg, path):
    paths = {}
    if cfg.base.dataset in ['kaggle_224', 'kaggle_512' ]: 
        if socket.gethostname() in hostname:
            paths['root'] = path.dset.root
            paths['dset_dir'] = path.dset.dir_512
        else:
            NotImplementedError('Not implemented Hostname')            
    else:
        raise ValueError('Not implemented dataset.')
    print('hostname: ', socket.gethostname())
    #print('root dataset dir: ', paths['root'])
    #print('dataset dir: ', paths['dset_dir'])
    return munchify(paths)

#------------------------
def load_dataset():
    '''
        Load and return the test and validation dataset into a pandas dataframe
    '''
    root = "/gpfs01/berens/data/data/DR/Kaggle_DR_resized/"
    test_file_name = 'kaggle_gradable_test_new_qual_eval.csv'
    val_file_name  = 'kaggle_gradable_val_new_qual_eval.csv'

    df_test = pd.read_csv(os.path.join(root, test_file_name))
    df_val  = pd.read_csv(os.path.join(root, val_file_name))
    return df_test, df_val

#------------------------
def load_conf_file(config_file, path):
    '''
        Load the conf file containing all the parameters with the path file containing the data path
    '''
    with open(config_file) as fhandle:
        cfg = yaml.safe_load(fhandle)
    
    with open(path) as fhandle:
        paths = yaml.safe_load(fhandle)
        
    cfg = munchify(cfg)
    paths = munchify(paths)
    cfg.data_paths = data_path(cfg, paths)
    return cfg

#------------------------
def load_resnet(cfg, resnet_weights):
    '''
        Load the resnet model with the saving weights for inference
    '''
    num_class = cfg.data.num_classes
    resnet_checkpoint = os.path.join(resnet_weights) 
    res_model = models.resnet50(pretrained=True)
    res_model.fc = nn.Linear(res_model.fc.in_features, num_class)
    res_model = res_model.to(cfg.base.device)

    weights = torch.load(resnet_checkpoint)
    res_model.load_state_dict(weights, strict=True)
    return res_model.eval()

#------------------------
def load_bagnet(cfg, bagnet_weights, baseline=False):
    ''''
        Load the bagnet model with pretrained parameters for inference
    '''
    num_class = cfg.data.num_classes
    if baseline:       
        model = pytorchnet.bagnet33(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_class) 
    else:
        model = generate_model(cfg)
    
    model = model.to(cfg.base.device)
    weights = torch.load(bagnet_weights)
    model.load_state_dict(weights, strict=True)   
    return model.eval()

#######
####### Load images and transformations (test and plots) ################
def load_EyePAC(path, df, tensor=False):
    '''
    return the corresponding image (tensor or numpy) from the dataframe: => (batch_size, C, H, W).

        Parameters:
            - path (str): image path location
            - df: dataframe
        
        Returns:
            - numpy array image range in [0, 1] with shape (b, C,H,W)  
            - PIL image
    '''
    file_paths = [os.path.join(path, file) for file in df]
    image_list = []    
    
    if tensor:
        pil_img = [Image.open(file) for file in file_paths]
        #pil_img = torch.stack(pil_img)
        return pil_img
    else: # numpy output           
        for file in file_paths:
            with Image.open(file) as image:
                image = np.asarray(image)              # comvert the PIL image into numpy array
                image = image / 255.                   # normalize the value between [0, 1]
                image = image.transpose([2,0,1])       # from numpy shape (H,W,C) to tensor shape (C,H,W)    
                image = np.expand_dims(image, axis=0)  # adding the batch size dimension: from (C,H,W) to (batch_size, C,H,W) 
                
            image_list.append(image)
            np_image = np.concatenate(image_list, axis=0)
        return np_image 

#######
#------------------------
def plot_transform(cfg, img, tensor=False):
    '''
    apply plot transformations on the images
    
        inputs:
            - img: PIL / np array (bs,C,H,W)
        
        outputs:
            - numpy array or tensor image range in [0, 1] with shape (b,H,W,C)  
    '''
    if tensor:
        transform = transforms.Compose([
            transforms.Resize((cfg.data.input_size)),
            transforms.ToTensor(),    
            transforms.CenterCrop(cfg.data.input_size)
            ])
        imgs = transform(img)
        imgs = torch.unsqueeze(imgs, dim=0).permute(0,2,3,1)  # add the batch dimension and permute dimension to fit numpy image shape
        imgs = imgs.numpy()
    else:
        transform = transforms.Compose([
            transforms.Resize((cfg.data.input_size)),
            #transforms.ToTensor(),                   # img is already in the range [0, 1] so no need to apply ToTensor
            transforms.CenterCrop(cfg.data.input_size)
            ])
        imgs = torch.from_numpy(img)                   # from numpy to tensor        
        imgs = transform(imgs)
        imgs = imgs.numpy().transpose([0,2,3,1]) #(bs,H,W,C) => convert the image back to numpy format after transformations for ploting
    return imgs

#------------------------
def test_transform(cfg, img, pil_tensor=False):
    '''
    apply the test transformations on the input for inference
        
        inputs:
            - np_im / pil_im : img -> (bs,C,H,W)
            
        outputs:
            - toch.Tensor: (bs,C,H,W)  
    '''
    if pil_tensor:
        transform = transforms.Compose([
                    transforms.Resize((cfg.data.input_size)),
                    transforms.ToTensor(),    
                    transforms.CenterCrop(cfg.data.input_size),
                    transforms.Normalize(cfg.data.mean, cfg.data.std)])
        imgs = transform(img)
        imgs = torch.unsqueeze(imgs, dim=0)
    else:
        transform = transforms.Compose([
                    transforms.Resize((cfg.data.input_size)),
                    transforms.CenterCrop(cfg.data.input_size),
                    transforms.Normalize(cfg.data.mean, cfg.data.std)])
        imgs = torch.from_numpy(img)  # from numpy to tensor
        imgs = transform(imgs)
    return imgs
