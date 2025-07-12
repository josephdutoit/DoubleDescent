import torch 
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, Dataset, TensorDataset
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CategoricalLabelNoiseTransform:
    def __init__(self, label_noise: float):
        self.label_noise = label_noise

    def __call__(self, label):
        if np.random.rand() < self.label_noise:
            new_label = np.random.randint(0, 10)
            while new_label == label:
                new_label = np.random.randint(0, 10)
            label = new_label
        return label
    
class RegressionLabelNoiseTransform:   
    def __init__(self, label_noise: float):
        self.label_noise = label_noise

    def __call__(self, label):
        noise = np.random.normal(0, self.label_noise)
        label = label + noise
        return label

class NoisedDataset(Dataset):
    def __init__(
            self, 
            dataset: str, 
            root: str, 
            train: bool, 
            label_noise: float | None = None,
            num_train: int | None = None,
            num_test: int | None = None,
            split: float | None = None,
        ):

        if split is not None and (num_train is not None or num_test is not None):
            raise ValueError("Cannot specify both split and num_train/num_test.")
        
        if dataset == 'cifar10':
            if num_train is not None or num_test is not None or split is not None:
                raise ValueError(
                    "num_train, num_test, and split are not applicable for CIFAR-10."
                )
            self.dataset = torchvision.datasets.CIFAR10(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
            self.label_transform = CategoricalLabelNoiseTransform(label_noise)
        elif dataset == 'cifar100':
            if num_train is not None or num_test is not None or split is not None:
                raise ValueError(
                    "num_train, num_test, and split are not applicable for CIFAR-100."
                )
            self.dataset = torchvision.datasets.CIFAR100(
                root=root, train=train, download=True, transform=transforms.ToTensor()
            )
            self.label_transform = CategoricalLabelNoiseTransform(label_noise)
        elif dataset == 'california_housing':
            housing = fetch_california_housing()
            X, y = housing.data, housing.target

            if split is not None:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=1-split, random_state=42
                )
            elif num_train is not None and num_test is not None:
                if num_train + num_test > len(X):
                    raise ValueError("num_train + num_test exceeds dataset size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=num_train, test_size=num_test, random_state=42
                )
            elif num_train is not None:
                if num_train > len(X):
                    raise ValueError("num_train exceeds dataset size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=num_train, test_size=len(X) - num_train, random_state=42
                )
            elif num_test is not None:
                if num_test > len(X):
                    raise ValueError("num_test exceeds dataset size.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=len(X) - num_test, test_size=num_test, random_state=42
                )
            else:
                raise ValueError("Could not specify train/test split. Either split, num_train, or num_test must be specified.")
            
            scaler = StandardScaler()
            if train:
                X_scaled = scaler.fit_transform(X_train)
                y_data = y_train
            else:
                X_scaled = scaler.fit_transform(X_train) 
                X_scaled = scaler.transform(np.vstack((X_train, X_test)))
                y_data = np.concatenate((y_train, y_test))
            
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_data).unsqueeze(1) 
            
            self.dataset = TensorDataset(X_tensor, y_tensor)
            self.label_transform = RegressionLabelNoiseTransform(label_noise)

        else:
            raise ValueError(f"Dataset {dataset} is not supported.")
        

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        label = self.label_transform(label)
        return image, label


def get_datasets(
        dataset: str, 
        root: str, 
        label_noise: float | None = None,
        num_train: int | None = None,
        num_test: int | None = None,
        split: float | None = None,
    ) -> tuple[NoisedDataset, NoisedDataset]:

    if dataset not in ['cifar10', 'cifar100', 'california_housing']:
        raise ValueError(f"Dataset {dataset} is not supported.")

    trainset = NoisedDataset(
        dataset=dataset,
        root=root,
        train=True,
        label_noise=label_noise,
        num_train=num_train,
        num_test=num_test,
        split=split
    )

    testset = NoisedDataset(
        dataset=dataset,
        root=root,
        train=False,
        label_noise=label_noise,
        num_train=num_train,
        num_test=num_test,
        split=split
    )

    return trainset, testset

def get_dataloaders(
        dataset: str,
        root: str,
        label_noise: float | None = None,
        num_train: int | None = None,
        num_test: int | None = None,
        split: float | None = None,
        batch_size: int = 128,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:

    trainset, testset = get_datasets(
        dataset=dataset, 
        root=root,  
        label_noise=label_noise, 
        num_train=num_train, 
        num_test=num_test, 
        split=split
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )

    return trainloader, testloader