import os
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import torch
from PIL import Image

class FundusDataset(Dataset):
    def __init__(self, cfg, train=True, test=False, transform=None):
        self.img_size = (cfg.data.input_size, cfg.data.input_size)
        self.aug = cfg.data.augmentation
        self.transform = transform
        self.image_path = os.path.join(cfg.data_paths.root, cfg.data_paths.dset_dir)
        csv_path = cfg.data_paths.root

        if cfg.data.binary:
            self.n_classes = 2
            self.str_label = cfg.data.onset
        else:
            self.n_classes = 5
            self.str_label = 'level'

        if not test: 
            if train:
                self.df = pd.read_csv(os.path.join(csv_path, cfg.berens.train_csv))
            else:
                self.df = pd.read_csv(os.path.join(csv_path, cfg.berens.val_csv))
        else:
            self.df = pd.read_csv(os.path.join(csv_path, cfg.berens.test_csv))

        self.filenames = self.df['filename']
        self.labels = self.df[self.str_label]
        self.classes = sorted(list(set(self.targets)))
        
        if cfg.base.test:            
            n=cfg.base.sample
            self.filenames = self.filenames[:n]
            self.labels = self.labels[:n]
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = os.path.join(self.image_path, self.filenames[idx])        
        if self.aug == 'murat':
            image = cv2.imread(filename)
            image = cv2.resize(image, self.img_size)
        else:
            image = Image.open(filename) # generate PIL image

        if self.transform:
            image = self.transform(image)
            
        label = self.labels.iloc[idx]

        return image, label
    
    def balanced_weights(self):

        #class_prop = [0.3, 0.7]    
        # class_weights = (1./len(class_weights)) / class_weights
        weights = [0] * len(self)

        for idx, val in enumerate(getattr(self.df, self.str_label)):
                weights[idx] = 1 / self.class_proportions[val]
        return weights

    @property
    def class_proportions(self):
        y = self.targets.view(-1, 1)
        targets_onehot = (y == torch.arange(self.n_classes).reshape(1, self.n_classes)).float()
        proportions = torch.div(torch.sum(targets_onehot, dim=0) , targets_onehot.shape[0])
        return proportions

    @property
    def targets(self):
        return torch.tensor(self.df[self.str_label].values)


## dataset for /mnt/qb/datasets/STAGING/berens/kaggle_dr_preprocessed
class FundusDataset_1024(VisionDataset):
    def __init__(self, cfg, root='', train=True, test=False, transform=None, target_transform=None):
        super(FundusDataset_1024, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.aug = cfg.data.augmentation
        if not test:
            self.image_path = cfg.paths['data_train_path']            
            select_side = cfg.berens.select_side

            assert select_side in ['left', 'right', None], "Invalid input"
            # create splits
            train_csv, val_csv = cfg.paths['train_csv_file'], cfg.paths['val_csv_file']
            df = pd.read_csv(train_csv) if train else pd.read_csv(val_csv)
            df['patient_id'] = df.apply(lambda row: row['image'].split('_')[0], axis=1)
            df['side'] = df.apply(lambda row: row['image'].split('_')[1], axis=1)
            if select_side:
                self.metadata = df[df['image'].str.contains(select_side)].reset_index()
            else:
                self.metadata = df

            if train:
                self.metadata.to_csv(os.path.join(cfg.paths['log_path'], 'final_train.csv'), index=False, columns=['image', 'level', 'side'])
            else:
                self.metadata.to_csv(os.path.join(cfg.paths['log_path'],'final_val.csv'), index=False, columns=['image', 'level', 'side'])
            #self.metadata = self.metadata[:30]
        else:
            self.image_path = cfg.paths['data_test_path'] 
            self.metadata = pd.read_csv(os.path.join(cfg.paths['root'], 'retinopathy_solution.csv'))
        
        self.classes = sorted(list(set(self.targets)))


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):   
        img_id = self.metadata['image'][idx]
        img_path = os.path.join(self.image_path, f"{img_id}.png")
        
        if self.aug == 'murat':
            img = cv2.imread(img_path)
        else:
            img = Image.open(img_path) # generate PIL image

        label = self.metadata['level'][idx]
       
        if self.transform is not None:
            img = self.transform(img)

        return (img, label)

    @property
    def targets(self):
        return torch.tensor(self.metadata['level'].values)
