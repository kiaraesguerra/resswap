# Create custom pytorch lightning callback to initialize a new model
# with the weights of a previous model
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from models.trial2 import trial2, resnet20

class ResSwap(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 10:
            print('ResSwap callback called')
            pl_module.model = resnet20(num_classes=10).to('cuda')
            pl_module.model.load_state_dict(torch.load(trainer.checkpoint_callbacks[0].best_model_path))
            
        
