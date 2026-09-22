from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import CheXpert_2_Dataset, CheXpert_Dataset
from medical_diffusion.models.embedders.latent_embedders import VAE
import os
from pytorch_lightning import loggers as pl_loggers

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def train_med_image():
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path("/data/coml-intersection-joins/lina4921/runs") / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None

    cache_dir = Path("/tmp") / os.environ.get("USER", "user") / "chexpert_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # tb_logger = pl_loggers.TensorBoardLogger(save_dir="./logs/")
    tb_logger = pl_loggers.CSVLogger("/data/coml-intersection-joins/lina4921/logs/", name="vae_chexpert")

    ds_3 = CheXpert_Dataset( #  256x256
        image_resize=128, 
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root = "/data/coml-intersection-joins/lina4921/chexpertchestxrays-u20210408/train_cheXbert.csv",
        use_cache=True,
        cache_dir=cache_dir
    )

    # ds = ConcatDataset([ds_1, ds_2, ds_3])
   
    # dm = SimpleDataModule(
    #     ds_train = ds_3,
    #     batch_size=8, 
    #     num_workers=4,
    #     pin_memory=True
    # ) 
    dm = SimpleDataModule(
        ds_train=ds_3,
        batch_size=32,          # increase until GPU memory is close to full
        num_workers=12,          # try 8 or 12 if CPU can keep up
        pin_memory=True,
        val_split=0.0           # no split if you are not validating
    )
    

    # ------------ Initialize Model ------------
    model = VAE(
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        hid_chs =    [ 64, 128, 256,  512], 
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,    2],
        deep_supervision=1,
        use_attention= 'none',
        loss = torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6
    )

    # model.load_pretrained(Path.cwd()/'runs/2022_12_01_183752_patho_vae/last.ckpt', strict=True)

    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 20

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        logger=tb_logger,
        precision="32-true",
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=1001,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)