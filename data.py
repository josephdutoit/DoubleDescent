import warnings
import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
import numpy as np


class NoisedDataset(Dataset):
    def __init__(
        self, 
        dataset: str, 
        root: str, 
        train: bool, 
        label_noise: float = 0.0,
    ):
        if dataset == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
            num_classes = 10
        elif dataset == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
            num_classes = 100
        else:
            raise ValueError(f"Dataset {dataset} is not supported.")

        if train and label_noise > 0:
            original_targets = np.array(self.dataset.targets)
            
            flip_mask = np.random.rand(len(self.dataset)) < label_noise
            
            noisy_targets = original_targets.copy()
            
            flip_indices = np.where(flip_mask)[0]
            for idx in flip_indices:
                original_label = original_targets[idx]
                new_label = np.random.randint(0, num_classes)
                while new_label == original_label:
                    new_label = np.random.randint(0, num_classes)
                noisy_targets[idx] = new_label
            
            self.dataset.targets = noisy_targets.tolist()
            

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        return image, label

def get_datasets(
        name: str,
        root: str = './datasets',
        download: bool = True,  
        label_noise: float = 0.0
    ) -> tuple[NoisedDataset, NoisedDataset]:

    
    if name not in ['cifar10', 'cifar100']:
        raise ValueError(f"Dataset {name} is not supported.")
    
    testset = NoisedDataset(
        dataset=name, root=root, train=False, label_noise=label_noise
    )
    trainset = NoisedDataset(
        dataset=name, root=root, train=True, label_noise=label_noise
    )
    

    return trainset, testset

def get_dataloaders(
        name: str,
        batch_size: int = 128,
        num_workers: int = 4,
        label_noise: float = 0.0
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    trainset, testset = get_datasets(name, label_noise=label_noise)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader