import os
import argparse
import pytorch_lightning as pl
from pathlib import Path
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import torch.utils.data as data

from loguru import logger

from models.harpnet.model import HARPNet

from data import Dataset
import yaml
from glob import glob
import numpy as np

import random


def run_testing(exp_path, save_path, ckpt = None, selected_tracks = None,
                batch_size=None, gpus=1, limit_test_batches=1.0, device='cpu', output_audio=True):
    
    exp_def = os.path.join(exp_path, 'hparams.yaml')
    
    # run through all ckpt files
    ckpt_paths = glob(os.path.join(exp_path, 'checkpoints','*.ckpt'))
    if ckpt is not None:
        ckpt_paths =  [p for p in ckpt_paths if ckpt in p]
    for ckpt_path in ckpt_paths:
        
        print('Testing on: {}'.format(ckpt_path))
        output_path = os.path.basename(ckpt_path).split('_')[1]

        default_hyperparams = yaml.safe_load(open(exp_def))
        default_hyperparams['audio_output_path'] = None
        
        # Set seed for random
        seed = np.random.randint(0,1000)
        print(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        if output_audio:
            default_hyperparams['audio_output_path'] = os.path.join(exp_path, output_path)
            Path(default_hyperparams['audio_output_path']).mkdir(parents=True, exist_ok=True)
        model = HARPNet.load_from_checkpoint(ckpt_path, default_params=default_hyperparams)
        model.hparams.batch_size = batch_size
        model.hparams.gpus = gpus

        model = model.to(device)
        model.eval()

        trainer = pl.Trainer(logger=False, checkpoint_callback=False, gpus=gpus,
                        default_root_dir=save_path,
                        limit_test_batches=limit_test_batches)
        
        test_dataset = Dataset(
            split="tt",
            root_in=default_hyperparams['data_dir_in'],
            root_out1=default_hyperparams['data_dir_out1'],
            root_out2=default_hyperparams['data_dir_out2'],
            chunk_size=default_hyperparams['H'],
            random_chunks=True,
            sample_rate=default_hyperparams['sr'],
            num_chan=default_hyperparams['in_channels'],
            scaler=default_hyperparams['scaler'],
            training=False
        )
        
        if selected_tracks is not None:
            test_dataset = data.Subset(test_dataset, selected_tracks)
            
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, 
                                                      num_workers=4)

        logger.add(os.path.join(os.path.join(*ckpt_path.split('/')[:-2]), 'test.log'), 
                backtrace=True, diagnose=True, rotation=None) 

        trainer = pl.Trainer(limit_test_batches=limit_test_batches,
                            callbacks=[],
                            gpus=gpus,
                            deterministic=True,
                            log_every_n_steps=50)

        logger.info('Starting the Testing!')
        trainer.test(model, test_dataloaders=test_dataloader)
        logger.info('Testing complete')

def main(args):
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    run_testing(args.exp_path, args.save_path, ckpt=args.ckpt_path, batch_size=args.batch_size, 
                selected_tracks=args.selected_tracks, gpus=args.gpus, limit_test_batches=args.limit_test_batches, device=device, 
                output_audio=args.output_audio)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--output-audio', type=bool, default=False)
    parser.add_argument('--save-path', type=str, default="None")
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--limit-test-batches', type=float, default=1.0)
    parser.add_argument('--selected-tracks', type=list, default=None)

    main(parser.parse_args())