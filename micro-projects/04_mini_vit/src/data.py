import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


def make_loader(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), transform=transform
    )
    
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "valid"), transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        drop_last=True
    )
    
    return train_loader, test_loader