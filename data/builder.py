from argparse import ArgumentError
import os
import pickle

from torchvision import datasets

from .loader import pil_loader
from .transforms import data_transforms, simple_transform, our_transform
from utils.func import mean_and_std, print_dataset_info
from .our_dataset import FundusDataset


def generate_dataset(cfg):
    
    if cfg.data.mean == 'auto' or cfg.data.std == 'auto':
        data_path = os.path.join(cfg.data_paths.root, cfg.data_paths.dset_dir)
            
        mean, std = auto_statistics(
            data_path, 
            cfg.base.data_index,
            cfg.data.input_size,
            cfg.train.batch_size,
            cfg.train.num_workers,
            cfg
        )        
        cfg.data.mean = mean  
        cfg.data.std = std 
    else:
        cfg.data.mean = cfg.data_paths.mean  
        cfg.data.std = cfg.data_paths.std 

    print('auto mean')
    if cfg.data.augmentation == 'other':
        train_transform, test_transform = our_transform(cfg)
    else:
        train_transform, test_transform = data_transforms(cfg) # baseline

    if 'kaggle' in cfg.base.dataset:
        print('kaggle......')
        datasets = generate_our_dataset_kaggle(
            cfg,
            train_transform,
            test_transform
        )
    elif cfg.base.dataset == 'eyepacs':
        print('eyepacs......')
        datasets = generate_our_dataset_kaggle( #generate_dataset_eyepacs
            cfg,
            train_transform,
            test_transform
        )
    else:
        raise ArgumentError(f'Dataset not implemented: {cfg.base.dataset}')

    print_dataset_info(datasets)
    return datasets


def auto_statistics(data_path, data_index, input_size, batch_size, num_workers, cfg):
    print('Calculating mean and std of training set for data normalization.')
    transform = simple_transform(input_size)

    if data_index not in [None, 'None']:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transform)
    else:
        ''' generate the training dataset'''
        #os.path.join(data_path, 'train') # initial 
        #train_dataset = datasets.ImageFolder(train_path, transform=transform)
        train_dataset = FundusDataset(cfg, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def generate_our_dataset_kaggle(cfg, train_transform, test_transform):
                
    dset_train = FundusDataset(cfg, transform=train_transform)
    dset_val = FundusDataset(cfg, train=False, transform=test_transform)
    dset_test = FundusDataset(cfg, train=False, test=True, transform=test_transform)

    return dset_train, dset_test, dset_val