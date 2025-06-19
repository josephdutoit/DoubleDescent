import lightning as pl
from torch import nn, optim
from models.resnet18k import make_resnet18k

class Model(pl.LightningModule):
    def __init__(self, cfg, width):
        super(Model, self).__init__()
        self.cfg = cfg
        self.width = width
        self.model = make_resnet18k(
            k=width,
            num_classes=cfg.num_classes
        )

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

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
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        aliasing_norm, data_insufficiency = self.model.get_aliasing_metrics()
        self.log('aliasing_norm', aliasing_norm, on_step=False, on_epoch=True)
        self.log('data_insufficiency', data_insufficiency, on_step=False)
