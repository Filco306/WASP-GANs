import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

data_set = datasets.ImageFolder("data", transform=transforms.ToTensor())
image_loader = DataLoader(data_set, batch_size=4)

psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for inputs in tqdm(image_loader):
    inputs, _ = inputs
    psum    += inputs.sum(axis= [0, 2, 3])
    psum_sq += (inputs ** 2).sum(axis= [0, 2, 3])


count = len(data_set) * 450 * 337

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))


torch.save({"mean": total_mean, "std": total_std}, "data_properties.pt")