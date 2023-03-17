import argparse as ap
import numpy as np
import logging
import os
import pathlib
import sys

import atom3d.datasets as da
import pytorch_lightning as pl
import pytorch_lightning.loggers as log
from pytorch_lightning.callbacks import ModelCheckpoint
import torch_geometric
import wandb

import edn.data as d
import edn.model as m


root_dir = pathlib.Path(__file__).parent.parent.absolute()
logger = logging.getLogger("lightning")
#wandb.init(project="edn")


def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('val_dataset', type=str)
    parser.add_argument('-out', '--output_dir', type=str, default='./output')
    parser.add_argument('-f', '--filetype', type=str, default='lmdb',
                        choices=['lmdb', 'pdb', 'silent'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--random_seed', '-seed', type=int, default=int(np.random.randint(1, 10e6)))

    # add model specific args
    parser = m.EDN_PL.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    logger.info(f"Set random seed to {hparams.random_seed:}...")
    pl.seed_everything(hparams.random_seed, workers=True)

    logger.info(f"Write output to {hparams.output_dir}")
    os.makedirs(hparams.output_dir, exist_ok=True)

    transform = d.EDN_Transform(True, num_nearest_neighbors=40)

    # DATA PREP
    logger.info(f"Creating dataloaders...")
    train_dataset = da.LMDBDataset(hparams.train_dataset, transform=transform)
    train_dataloader = torch_geometric.data.DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        shuffle=True)
    val_dataset = da.LMDBDataset(hparams.val_dataset, transform=transform)
    val_dataloader = torch_geometric.data.DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    # Initialize model
    logger.info("Initializing model...")
    edn_pl = m.EDN_PL(**dict_args)
    print(edn_pl.net)

    #wandb_logger = log.WandbLogger(save_dir=hparams.output_dir)
    tb_logger = pl.loggers.TensorBoardLogger(hparams.output_dir, name="tensorboard", version="")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(hparams.output_dir, 'checkpoints'),
        filename='edn-{epoch:03d}-{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        )
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        logger=tb_logger,
        callbacks=[checkpoint_callback],
        )

    # TRAINING
    logger.info(f"Running training on {hparams.train_dataset:} with val {hparams.val_dataset:}...")
    out = trainer.fit(edn_pl, train_dataloader, val_dataloader)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()

