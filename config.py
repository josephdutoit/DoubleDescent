import json
from dataclasses import dataclass

@dataclass
class Config:
    net_type: str = 'resnet18'
    dataset: str = 'cifar10'
    num_classes: int = 10
    label_noise: float = 0.2
    optimizer: str = 'adam'
    max_steps: int = 500000
    max_epochs: int = 4000
    lr: float | None = 1e-4
    batch_size: int = 128
    log_dir: str = './logs'

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