# %%
import torch.utils.data
import random
import torch
import os
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from loguru import logger

from models.harpnet.model import HARPNet
from models.harpnet.modules import SNREntropyCheckpoint

from data import Dataset

def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def load_datasets(args):

        train_dataset = Dataset(
            split="tr",
            root_in=args.data_dir_in,
            root_out1=args.data_dir_out1,
            root_out2=args.data_dir_out2,
            chunk_size=args.H,
            random_chunks=True,
            sample_rate=args.sr,
            num_chan=args.in_channels,
            scaler=args.scaler,
            training=True
        )

        valid_dataset = Dataset(
            split="cv",
            root_in=args.data_dir_in,
            root_out1=args.data_dir_out1,
            root_out2=args.data_dir_out2,
            chunk_size=args.H,
            random_chunks=True,
            sample_rate=args.sr,
            num_chan=args.in_channels,
            scaler=args.scaler,
            training=False
        )

        return train_dataset, valid_dataset

    train_dataset, valid_dataset = load_datasets(args)
    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": False} if torch.cuda.is_available() else {}
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=True, 
                                                   drop_last=True,
                                                   **dataloader_kwargs)
    val_dataloader = torch.utils.data.DataLoader(valid_dataset, 
                                                 batch_size=args.batch_size,
                                                 drop_last=True, 
                                                 **dataloader_kwargs)

    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Model Instantiate
    model = HARPNet(**vars(args))
    model = model.to(device)
    import pdb; pdb.set_trace()

    # Append the job id as a suffix
    job_id = os.getpid()
    version = '{}_{}'.format(args.save_version, job_id)
    
    logger.add(os.path.join(args.save_path, args.save_name, version, 'train.log'), backtrace=True, diagnose=True) 
    logger.info('Starting the Training!')

    #Logger
    log = TensorBoardLogger(save_dir=args.save_path, name=args.save_name,
                                version=version)
    
    # Checkpoint callback
    ckpt_delay = 0
    ckpt_callback = SNREntropyCheckpoint(monitor_delay=ckpt_delay, 
                                         monitor_entropy=args.loss_weights['entropy']>0.0, overall_entropy=args.baseline,
                                         dirpath=os.path.join(args.save_path, args.save_name, version, 'checkpoints'))
    
    # Trainer Instantiate
    trainer = pl.Trainer(min_epochs=args.epochs, max_epochs=args.epochs,
                        logger=log,
                        enable_checkpointing=False,
                        resume_from_checkpoint=args.checkpoint,
                        reload_dataloaders_every_epoch=True,
                        limit_train_batches=1.0,
                        limit_val_batches=1.0,
                        limit_test_batches=1.0,
                        callbacks=[ckpt_callback],
                        gpus=1, num_nodes=1,
                        progress_bar_refresh_rate=0,
                        deterministic=False,
                        log_every_n_steps=50,
                        detect_anomaly=True,
                        accumulate_grad_batches=args.acc_grad)
    
    # Train model
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    logger.info('Training complete')

if __name__ == '__main__':
    
    import yaml
    from argparse import Namespace
    import argparse 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-def', type=str, default="./confs/conf_mb.yml")
    args = parser.parse_args()
    
    with open(args.exp_def) as f:
        def_conf = yaml.safe_load(f)
        
    hparams = Namespace(**def_conf)
    main(hparams)