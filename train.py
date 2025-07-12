import argparse
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from model import Model
from data import get_dataloaders
from config import Config, main


def main(
        cfg: Config, 
        model_width: int | None = None,
        num_samples: int | None = None,
        gpu_id: int | None = None,
        num_workers: int = 4
    ):
    
    train_loader, test_loader = get_dataloaders(
        dataset=cfg.dataset,
        label_noise=cfg.label_noise,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        num_train=num_samples,
        root=cfg.data_dir
    )

    logger = TensorBoardLogger(
        save_dir=cfg.log_dir,
        name=f'{cfg.net_type}_{model_width}',
        default_hp_metric=False
    )

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=cfg.max_steps,
        logger=logger,
        accelerator='gpu',
        devices=[gpu_id],
        enable_progress_bar=True,
        log_every_n_steps=cfg.log_freq,
        check_val_every_n_epoch=cfg.val_freq,
    )

    model = Model(cfg, width=model_width)

    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet model with specified width.')
    parser.add_argument(
                '--width', 
                type=int, 
                required=False,
                help='Width of the ResNet model'
    )

    parser.add_argument(
        '--train_samples',
        type=int,
        required=False,
        help='Number of training samples to use'
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

    main(
        cfg=cfg, 
        model_width=args.width, 
        num_samples=args.train_samples, 
        gpu_id=args.gpu_id, 
        num_workers=args.num_workers
    )
