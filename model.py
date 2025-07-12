import lightning as pl
from torch import nn, optim
from models.resnet18k import make_resnet18k
from models.lin_reg import LinearRegressionModel

class Model(pl.LightningModule):
    def __init__(self, cfg, width):
        super(Model, self).__init__()
        self.cfg = cfg
        self.width = width

        if cfg.net_type == 'resnet18k':
            self.model = make_resnet18k(
                k=width,
                num_classes=cfg.num_classes
            )
        elif cfg.net_type == 'lin_reg':
            self.model = LinearRegressionModel(
                num_samples=width,
                feature_dim=cfg.feature_dim,
            )
        else:
            raise ValueError(f"Unsupported network type: {cfg.net_type}")

        if cfg.loss_fn == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {cfg.loss_fn}")

        self.save_hyperparameters(cfg.__dict__)

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        if self.cfg.optimizer == 'adam':
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.cfg.lr
            )
        elif self.cfg.optimizer == 'sgd':
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.cfg.lr, 
                momentum=0.9, 
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.cfg.optimizer}")
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.criterion(out, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)
        aliasing_norm, data_insufficiency = self.model.get_aliasing_metrics()
        self.log('aliasing_norm', aliasing_norm, on_step=True, on_epoch=False)
        self.log('data_insufficiency', data_insufficiency, on_step=True, on_epoch=False)
