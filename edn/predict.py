import argparse as ap
import logging
import os
import sys
from pathlib import Path

import atom3d.datasets as da
import dotenv as de
import pandas as pd
import pathlib
import pytorch_lightning as pl
import torch_geometric

import edn.data as d
import edn.model as m


root_dir = pathlib.Path(__file__).parent.absolute()
de.load_dotenv(os.path.join(root_dir, '.env'))
logger = logging.getLogger("lightning")


def main():
    parser = ap.ArgumentParser()
    # add PROGRAM level args
    parser.add_argument('dataset', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('output_file', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--nolabels', dest='use_labels', action='store_false')
    parser.add_argument('--num_workers', type=int, default=1)

    # add model specific args
    parser = m.EDN_PL.add_model_specific_args(parser)

    # add trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    dict_args = vars(hparams)

    transform = d.EDN_Transform(hparams.use_labels, num_nearest_neighbors=40)

    # DATA PREP
    logger.info(f"Setup dataloaders...")
    dataset = da.LMDBDataset(hparams.dataset, transform=transform)
    dataloader = torch_geometric.data.DataLoader(
        dataset,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers)

    # MODEL SETUP
    logger.info("Loading model weights...")
    edn_pl = m.EDN_PL.load_from_checkpoint(hparams.checkpoint_path)

    trainer = pl.Trainer.from_argparse_args(hparams)

    # PREDICTION
    logger.info("Running prediction...")
    out = trainer.test(edn_pl, dataloader, verbose=False)

    # SAVE OUTPUT
    os.makedirs(Path(os.path.abspath(hparams.output_file)).parent, exist_ok=True)
    logger.info(f"Saving prediction to {hparams.output_file}...")
    df = pd.DataFrame(edn_pl.predictions)
    df.to_csv(hparams.output_file, index=False, float_format='%.7f')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout,
                        format='%(asctime)s %(levelname)s %(process)d: ' +
                        '%(message)s',
                        level=logging.INFO)
    main()

