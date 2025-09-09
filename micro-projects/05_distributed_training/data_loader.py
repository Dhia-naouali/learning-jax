import os
import numpy as np

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def make_loader(batch_size, train=True, num_workers=None):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else T.Identity(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    dataset = datasets.CIFAR10(root="data", download=True, train=train, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=os.cpu_count() if num_workers is None else num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return loader

def transform_batch(batch):
    images, labels = batch
    images = images.detach().cpu().numpy().astype(np.float32)
    labels = labels.detach().cpu().numpy().astype(np.int32)
    images = np.transpose(images, (0, 2, 3, 1))
    return images, labels
