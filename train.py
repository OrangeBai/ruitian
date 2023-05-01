import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from core.trainer import PLModel

if __name__ == '__main__':

    with open('config.yaml', 'r') as f:
        config = yaml.full_load(f)

    logtool = WandbLogger(name=config['name'], save_dir=config['model_dir'], project='Ruitian')

    model = PLModel(config)

    callbacks = [
        ModelCheckpoint(monitor='val/loss', save_top_k=1, mode="min", save_on_train_epoch_end=False,
                        dirpath=logtool.experiment.dir, filename="best")
    ]

    trainer = pl.Trainer(devices="auto",
                         precision=32,
                         amp_backend="native",
                         accelerator="cuda",
                         strategy='dp',
                         callbacks=callbacks,
                         max_epochs=config['train']['num_epoch'],
                         logger=logtool,
                         inference_mode=False,
                         accumulate_grad_batches=model.grad_acc,
                         )
    trainer.fit(model)
    # model.save_model(logtool.experiment.dir)
