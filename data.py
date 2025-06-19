import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset
import numpy as np

class LabelNoiseTransform:
    def __init__(self, label_noise: float):
        self.label_noise = label_noise

    def __call__(self, label):
        if np.random.rand() < self.label_noise:
            new_label = np.random.randint(0, 10)
            while new_label == label:
                new_label = np.random.randint(0, 10)
            label = new_label
        return label

class NoisedDataset(Dataset):
    def __init__(self, dataset: str, root: str, train: bool, label_noise: float = 0.0):
        if dataset == 'cifar10':
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
        elif dataset == 'cifar100':
            self.dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
        else:
            raise ValueError(f"Dataset {dataset} is not supported.")

        self.label_transform = LabelNoiseTransform(label_noise)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        label = self.label_transform(label)
        return image, label

def get_datasets(
        name: str,
        root: str = './datasets',
        download: bool = True,  
        label_noise: float = 0.0
    ) -> tuple[NoisedDataset, NoisedDataset]:

    
    if name not in ['cifar10', 'cifar100']:
        raise ValueError(f"Dataset {name} is not supported.")
    
    trainset = NoisedDataset(
        dataset=name, root=root, train=True, label_noise=label_noise
    )
    testset = NoisedDataset(
        dataset=name, root=root, train=False, label_noise=label_noise
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