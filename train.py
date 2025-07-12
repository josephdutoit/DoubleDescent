import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from model import Model
from data import get_dataloaders
from config import Config, main


def main(
        cfg: Config, 
        model_width: int,
        gpu_id: int | None = None,
        num_workers: int = 4
    ):

    # device = f'cuda:{gpu_id}' if gpu_id is not None else 'cuda'
    
    train_loader, test_loader = get_dataloaders(
        name=cfg.dataset,
        label_noise=cfg.label_noise,
        num_workers=num_workers,
        batch_size=cfg.batch_size
    )

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=f'{cfg.net_type}_{model_width}',
        default_hp_metric=False
    )

    gpus = [gpu_id] if gpu_id is not None else "auto"
    max_steps = cfg.max_steps if cfg.max_steps is not None else -1
    
    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=max_steps,
        logger=logger,
        accelerator='gpu',
        devices=gpus,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,  # Validate every epoch
        num_sanity_val_steps=0,     # Disable sanity checks
    )

    model = Model(cfg, width=model_width)

    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet model with specified width.')
    parser.add_argument(
                '--width', 
                type=int, 
                required=True,
                help='Width of the ResNet model'
    )
    
    parser.add_argument(
                '--config', 
                type=str, 
                required=True,
                default='resnet18k_config',
                help='Path to the configuration file'
    )
    
    parser.add_argument(
                '--gpu_id',
                type=int,
                default=None,
                help='GPU ID to use for training (default: None, uses all available GPUs)'
    )

    parser.add_argument(
                '--num_workers',
                type=int,
                default=4,
                help='Number of workers for data loading'
    )

    args = parser.parse_args()
    cfg = Config.load_from_file("configs/" + args.config + ".json")

    main(cfg, args.width, args.gpu_id, args.num_workers)
