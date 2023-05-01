import pytorch_lightning as pl
import pandas as pd
import os
from models.model_zoo import set_model
from core.torch_io import set_dataloader, Normalize
import torch
from utils.external import load_feat_uid
from utils.internal import init_optimizer, init_scheduler


class PLModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat, self.uid = self._set_feat_uid()
        self.input_uid, self.target_uid = self._set_target()

        self.train_loader = self._set_loader('train')
        self.val_loader = self._set_loader('test')

        self.norm_layer = self._set_norm_layer()
        self.loss_function = torch.nn.MSELoss()
        self.model = self._set_model()
        # self.trainer.accumulate_grad_batches = 1

    @property
    def grad_acc(self):
        if self.config['model_setting']['model'] == 'linear':
            # return len(self.train_loader)
            return self.config['train']['grad_acc']
        else:
            return self.config['train']['grad_acc']

    def _set_norm_layer(self):
        base_dir = self.config['dataset']['base']['base_dir']
        mean = pd.read_csv(os.path.join(base_dir, 'mean.csv'), index_col=0)
        std = pd.read_csv(os.path.join(base_dir, 'std.csv'), index_col=0)
        mean = torch.tensor(mean.loc[0, self.feat]).cuda()
        std = torch.tensor(std.loc[0, self.feat]).cuda()
        return Normalize(mean, std)

    def _set_feat_uid(self):
        config = self.config['dataset']['base']
        return load_feat_uid(config['base_dir'], config['folders'])

    def _set_target(self):
        target_uid = self.config['dataset']['base']['target_uid']
        input_uid = self.config['dataset']['base']['input_uid']
        return target_uid if target_uid is not None else self.uid, input_uid if input_uid is not None else self.uid

    def _set_loader(self, mode='train'):
        config = self.config['dataset']['base']
        config['target_uid'] = self.target_uid
        if mode == 'train':
            config['period'] = self.config['dataset']['train_period']
        else:
            config['period'] = self.config['dataset']['test_period']
        return set_dataloader(**config)

    def _set_model(self):
        model = self.config['model_setting']['model']
        base_config = {'num_feat': len(self.feat), 'num_uid': len(self.input_uid),
                       'num_output': len(self.target_uid)}

        if model in self.config['model_setting']:
            base_config.update(self.config['model_setting'][model])
        return set_model(model, **base_config)

    def setup(self, stage=None):
        if stage == 'fit':
            return
        else:
            return

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self, ):
        config = self.config['lr_opt']
        num_step = self.trainer.max_epochs * len(self.train_loader) / self.grad_acc
        opt = config['optimizer']
        opt_config = config[opt]
        optimizer = init_optimizer(self.model, optimizer=opt, lr=config['lr'], **opt_config)

        sch_name = config['scheduler']
        sch_config = config[sch_name]

        lr_scheduler = init_scheduler(sch_name, lr=config['lr'], num_step=num_step, optimizer=optimizer, **sch_config)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    @property
    def lr(self):
        return self.trainer.optimizers[0].param_groups[0]['lr']

    def training_step(self, batch, batch_idx):
        x, y = batch[0].cuda(), batch[1].cuda()
        x = self.norm_layer(x)
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        invalid_x = check_invalid(x)
        x[invalid_x] = 0
        invalid_y = check_invalid(y)
        y[invalid_y] = 0

        outputs = self.model(x)
        outputs[invalid_y] = 0

        loss = self.loss_function(outputs, y)
        self.log('lr', self.lr, sync_dist=True)
        self.log('train/loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0].cuda(), batch[1].cuda()
        x = self.norm_layer(x)
        x = x.to(torch.float32)
        y = y.to(torch.float32)

        invalid_x = check_invalid(x)
        x[invalid_x] = 0
        invalid_y = check_invalid(y)
        y[invalid_y] = 0

        outputs = self.model(x)
        outputs[invalid_y] = 0

        loss = self.loss_function(outputs, y)
        self.log('val/loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        return


    def save_model(self, save_dir):
        torch.save(self.model, os.path.join(save_dir, 'model.pth'))
        return


def check_invalid(x):
    return torch.any(torch.stack([x == 0, x.isnan(), x.isinf()]), dim=0)
