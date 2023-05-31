import argparse
import os
import time
import pickle
import torch
from data.moorfields import MoorfieldsDataset
import torchvision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--image_size', type=int, required=True)

def batch_mean_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)        
    return mean,std

if __name__ == "__main__":
    
    args = parser.parse_args()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.image_size),
        torchvision.transforms.ToTensor()
    ])
    dataset = MoorfieldsDataset(split="train", transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        #torch.utils.data.Subset(dataset, list(range(100))),
        batch_size=30,
        num_workers=8,
        shuffle=False,
    )
    
    mean, std = batch_mean_sd(loader)
    
    mean_std = {
        "mean": mean,
        "std": std,
    }
    
    directory = 'dataset/splits/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    with open(f'dataset/splits/mean_std_train_{args.image_size}.pkl', 'wb') as f:
        pickle.dump(mean_std, f)   