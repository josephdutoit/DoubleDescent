import json
from dataclasses import dataclass

@dataclass
class Config:
    # General Parameters
    seed: int = 42
    log_dir: str = './logs'
    data_dir: str = './data'
    val_freq: int = 1
    log_freq: int = 1
    plot_pbar: bool = True

    # Model Parameters
    net_type: str = 'resnet18'
    num_classes: int = 10
    feature_dim: int | None = None
    max_width: int = None

    # Training Parameters
    optimizer: str = 'adam'
    max_steps: int = None
    max_epochs: int = 4000
    lr: float | None = 1e-4
    loss_fn: str = 'cross_entropy'

    # Data Parameters
    dataset: str = 'cifar10'
    shuffle: bool = True
    split: float | None= None
    num_train: int | None = None
    num_test: int | None = None
    label_noise: float | None = None
    batch_size: int = 128

    @classmethod
    def load_from_file(self, file_path: str):
        with open(file_path, 'r') as file:
            config_data = json.load(file)
        return Config(**config_data)

    def save_to_file(self, file_path: str):
        with open(file_path, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

def main():
    cfg = Config()
    cfg.save_to_file('./configs/resnet18k_config.json')

if __name__ == '__main__':
    main()