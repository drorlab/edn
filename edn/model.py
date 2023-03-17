import argparse as ap
import collections as col
from functools import partial

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import torch_geometric as tg

from e3nn.kernel import Kernel
from e3nn.linear import Linear
from e3nn import o3
from e3nn.non_linearities.norm import Norm
from e3nn.non_linearities.nonlin import Nonlinearity
from e3nn.point.message_passing import Convolution
from e3nn.radial import GaussianRadialModel


class EDN_Model(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        # Define the input and output representations
        Rs0 = [(4, 0)]
        Rs1 = [(40, 0)]
        Rs20 = [(40, 0)]
        Rs21 = [(40, 1)]
        Rs22 = [(40, 2)]
        Rs3 = [(40, 0), (40, 1), (40, 2)]
        Rs30 = [(40, 0)]
        Rs31 = [(40, 1)]
        Rs32 = [(40, 2)]
        # To account for multiple output paths of conv.
        Rs30_exp = [(3 * 40, 0)]
        Rs31_exp = [(6 * 40, 1)]
        Rs32_exp = [(6 * 40, 2)]

        relu = torch.nn.ReLU()
        # Radial model:  R+ -> R^d
        RadialModel_1 = partial(
            GaussianRadialModel, max_radius=10.0, number_of_basis=20, h=12,
            L=1, act=relu)
        RadialModel_2 = partial(
            GaussianRadialModel, max_radius=20.0, number_of_basis=40, h=12,
            L=1, act=relu)

        ssp = ShiftedSoftplus()
        self.elu = torch.nn.ELU()

        # kernel: composed on a radial part that contains the learned
        # parameters and an angular part given by the spherical hamonics and
        # the Clebsch-Gordan coefficients
        selection_rule = partial(o3.selection_rule_in_out_sh, lmax=2)
        K1 = partial(
            Kernel, RadialModel=RadialModel_1, selection_rule=selection_rule)

        ### Layer 1
        self.lin1 = Linear(Rs0, Rs1)

        self.conv10 = Convolution(K1(Rs1, Rs20))
        self.conv11 = Convolution(K1(Rs1, Rs21))
        self.conv12 = Convolution(K1(Rs1, Rs22))

        self.norm = Norm()

        self.lin20 = Linear(Rs20, Rs20)
        self.lin21 = Linear(Rs21, Rs21)
        self.lin22 = Linear(Rs22, Rs22)

        self.nonlin10 = Nonlinearity(Rs20, act=ssp)
        self.nonlin11 = Nonlinearity(Rs21, act=ssp)
        self.nonlin12 = Nonlinearity(Rs22, act=ssp)

        ### Layer 2
        self.lin30 = Linear(Rs20, Rs30)
        self.lin31 = Linear(Rs21, Rs31)
        self.lin32 = Linear(Rs22, Rs32)

        def filterfn_def(x, f):
            return x == f

        self.conv2 = torch.nn.ModuleDict()
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    filterfn = partial(filterfn_def, f=f)
                    selection_rule = \
                        partial(o3.selection_rule, lmax=2, lfilter=filterfn)
                    K = partial(Kernel, RadialModel=RadialModel_2,
                                selection_rule=selection_rule)
                    self.conv2[str((i, f, o))] = \
                        Convolution(K([Rs3[i]], [Rs3[o]]))

        self.lin40 = Linear(Rs30_exp, Rs30)
        self.lin41 = Linear(Rs31_exp, Rs31)
        self.lin42 = Linear(Rs32_exp, Rs32)

        self.nonlin20 = Nonlinearity(Rs30, act=ssp)
        self.nonlin21 = Nonlinearity(Rs31, act=ssp)
        self.nonlin22 = Nonlinearity(Rs32, act=ssp)

        ### Final dense layers
        self.dense1 = torch.nn.Linear(40, 250, bias=True)
        self.dense2 = torch.nn.Linear(250, 150, bias=True)
        self.dense3 = torch.nn.Linear(150, 1, bias=True)

    def forward(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        ### Layer 1
        out = self.lin1(data.x)

        out0 = self.conv10(out, edge_index, edge_attr)
        out1 = self.conv11(out, edge_index, edge_attr)
        out2 = self.conv12(out, edge_index, edge_attr)

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.lin20(out0)
        out1 = self.lin21(out1)
        out2 = self.lin22(out2)

        out0 = self.nonlin10(out0)
        out1 = self.nonlin11(out1)
        out2 = self.nonlin12(out2)

        ### Layer 2
        out0 = self.lin30(out0)
        out1 = self.lin31(out1)
        out2 = self.lin32(out2)

        ins = {0: out0, 1: out1, 2: out2}
        tmp = col.defaultdict(list)
        for i in range(3):
            for f in range(3):
                for o in range(abs(f - i), min(i + f + 1, 3)):
                    curr = self.conv2[str((i, f, o))](
                        ins[i], edge_index, edge_attr)
                    tmp[o].append(curr)
        out0 = torch.cat(tmp[0], axis=1)
        out1 = torch.cat(tmp[1], axis=1)
        out2 = torch.cat(tmp[2], axis=1)

        # all atoms -> CAs
        CA_sel = torch.nonzero(data['select_ca'].squeeze(dim=0)).squeeze(dim=1)
        out0 = torch.squeeze(out0[CA_sel])
        out1 = torch.squeeze(out1[CA_sel])
        out2 = torch.squeeze(out2[CA_sel])
        # Also need to update the nodes/edges indexing
        edge_index, edge_attr = tg.utils.subgraph(CA_sel, edge_index, edge_attr, relabel_nodes=True)
        batch = torch.squeeze(data.batch[CA_sel])

        out0 = self.norm(out0)
        out1 = self.norm(out1)
        out2 = self.norm(out2)

        out0 = self.lin40(out0)
        out1 = self.lin41(out1)
        out2 = self.lin42(out2)

        out0 = self.nonlin20(out0)
        out1 = self.nonlin21(out1)
        out2 = self.nonlin22(out2)

        # Per-channel mean.
        out = scatter_mean(out0, batch, dim=0)

        out = self.dense1(out)
        out = self.elu(out)
        out = self.dense2(out)
        out = self.elu(out)
        out = self.dense3(out)
        out = torch.squeeze(out, axis=1)
        return out


class EDN_PL(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ap.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        return parser

    def __init__(self, learning_rate=1e-3, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.predictions = col.defaultdict(list)
        self.net = EDN_Model(**self.hparams)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.huber_loss(y_hat, batch.label.float())
        self.log('loss', loss, batch_size=len(batch))
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.huber_loss(y_hat, batch.label.float())
        self.log('val_loss', loss, prog_bar=True, batch_size=len(batch))
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = torch.nn.functional.huber_loss(y_hat, batch.label.float())
        self.predictions['id'].extend(batch.id)
        self.predictions['target'].extend(batch.label.cpu().numpy())
        self.predictions['pred'].extend(y_hat.cpu().numpy())
        return {'test_loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def forward(self, data):
        return self.net(data)


class ShiftedSoftplus:
    def __init__(self):
        self.shift = torch.nn.functional.softplus(torch.zeros(())).item()

    def __call__(self, x):
        return torch.nn.functional.softplus(x).sub(self.shift)

