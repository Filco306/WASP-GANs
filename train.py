r"""
train.py 

PyTorch-Lightning Trainer file, main file to run your experiments with
"""
import argparse
import os
import tensorboard
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.hparams import create_hparams
from src.data_module import MyDataModule
from src.training_model import DCGAN
# from run_tests import run_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    args = parser.parse_args()

    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise FileExistsError("Check point not present recheck the name")

    hparams = create_hparams()

    seed_everything(hparams.seed)

    data_module = MyDataModule(hparams)
    model = DCGAN(hparams, img_shape=(hparams.n_channels, *hparams.image_resized))

    # Callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.checkpoint_path,
                                          verbose=True,
                                          every_n_val_epochs=1)

    logger = TensorBoardLogger('tb_logs', name=hparams.run_name)

    trainer = pl.Trainer(resume_from_checkpoint=args.checkpoint_path,
                         default_root_dir=os.path.join("checkpoints", hparams.run_name),
                         gpus=hparams.gpus,
                         logger=logger,
                         log_every_n_steps=1,
                         flush_logs_every_n_steps=10,
                         #  plugins=DDPPlugin(find_unused_parameters=True),
                         #  accelerator='ddp',
                         check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                         gradient_clip_val=hparams.grad_clip_thresh,
                         callbacks=[checkpoint_callback],
                         track_grad_norm=2
                         )

    trainer.fit(model, data_module)
