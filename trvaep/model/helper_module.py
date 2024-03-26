import torch
import torch.nn as nn

from trvaep.utils import one_hot_encoder
import numpy as np


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_dim,
                 use_bn, use_dr, dr_rate, num_classes=None):
        super().__init__()
        self.n_classes = num_classes
        if num_classes is not None:
            layer_sizes[0] += num_classes
        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.FC.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
            if use_bn:
                self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
            self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if use_dr:
                self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_dim)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_dim)

    def forward(self, x, c=None):
        if c is not None:
            c = one_hot_encoder(c, n_cls=self.n_classes)
            x = torch.cat((x, c), dim=-1)
        x = self.FC(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_dim,
                 use_bn, use_dr, dr_rate, use_mmd=False, num_classes=None, output_active="linear"):
        super().__init__()
        self.use_mmd = use_mmd
        self.op_activation = output_active
        self.use_bn = use_bn
        self.use_dr = use_dr
        if num_classes is not None:
            self.n_classes = num_classes
            input_size = latent_dim + num_classes
        else:
            input_size = latent_dim
        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            if i + 1 < len(layer_sizes):
                self.FC.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                if self.use_bn:
                    self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if self.use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
            else:
                if self.op_activation == "ReLU":
                    self.FC.add_module(
                        name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module(name="output", module=nn.ReLU())
                if self.op_activation == "linear":
                    self.FC.add_module(name="output".format(i), module=nn.Linear(in_size, out_size))

    def forward(self, z, c=None):
        if c is not None:
            c = one_hot_encoder(c, n_cls=self.n_classes)
            z = torch.cat((z, c), dim=-1)
        x = self.FC(z)
        if self.use_mmd:
            y = self.FC.L0(z)
            if self.use_bn:
                y = self.FC.B0(y)
            y = self.FC.A0(y)
            if self.use_dr:
                y = self.FC.D0(y)
            return x, y
        return x
    
class Decoder_split_vars(nn.Module):

    def __init__(self, layer_sizes, latent_dim,
                 use_bn, use_dr, dr_rate, cols_split_idx, use_mmd=False, num_classes=None, output_active="linear"):
        super().__init__()
        self.use_mmd = use_mmd
        self.op_activation = output_active
        self.use_bn = use_bn
        self.use_dr = use_dr
        self.cols_split_idx = cols_split_idx #define last index of discrete columns
        if num_classes is not None:
            self.n_classes = num_classes
            input_size = latent_dim + num_classes
        else:
            input_size = latent_dim
        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            if i + 1 < len(layer_sizes):
                self.FC.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                if self.use_bn:
                    self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if self.use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
            else:
                if self.op_activation == "ReLU":
                    self.FC.add_module(
                        name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module(name="output", module=nn.ReLU())
                if self.op_activation == "linear":
                    self.FC.add_module(name="output".format(i), module=nn.Linear(in_size, out_size))

    def forward(self, z, c=None):
        if c is not None:
            c = one_hot_encoder(c, n_cls=self.n_classes)
            z = torch.cat((z, c), dim=-1)
        x = self.FC(z)
        #pass discrete columns through sigmoid, then concatenate back with continuous columns
        #first half of split is discrete cols, second half is continuous
        x = torch.cat((torch.sigmoid(x[:,:self.cols_split_idx]),x[:,self.cols_split_idx:]), dim=1)
        if self.use_mmd:
            y = self.FC.L0(z)
            if self.use_bn:
                y = self.FC.B0(y)
            y = self.FC.A0(y)
            if self.use_dr:
                y = self.FC.D0(y)
            return x, y
        return x
    
class Decoder_sigmoid_op(nn.Module):

    def __init__(self, layer_sizes, latent_dim,
                 use_bn, use_dr, dr_rate, use_mmd=False, num_classes=None, output_active="linear"):
        super().__init__()
        self.use_mmd = use_mmd
        self.op_activation = output_active
        self.use_bn = use_bn
        self.use_dr = use_dr
        if num_classes is not None:
            self.n_classes = num_classes
            input_size = latent_dim + num_classes
        else:
            input_size = latent_dim
        self.FC = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            if i + 1 < len(layer_sizes):
                self.FC.add_module(
                    name="L{:d}".format(i), module=nn.Linear(in_size, out_size, bias=False))
                if self.use_bn:
                    self.FC.add_module("B{:d}".format(i), module=nn.BatchNorm1d(out_size, affine=True))
                self.FC.add_module(name="A{:d}".format(i), module=nn.ReLU())
                if self.use_dr:
                    self.FC.add_module(name="D{:d}".format(i), module=nn.Dropout(p=dr_rate))
            else:
                if self.op_activation == "ReLU":
                    self.FC.add_module(
                        name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                    self.FC.add_module(name="output", module=nn.ReLU())
                if self.op_activation == "linear":
                    self.FC.add_module(name="output".format(i), module=nn.Linear(in_size, out_size))

    def forward(self, z, c=None):
        if c is not None:
            c = one_hot_encoder(c, n_cls=self.n_classes)
            z = torch.cat((z, c), dim=-1)
        x = self.FC(z)

        #pass discrete (all) columns through sigmoid
        x = torch.sigmoid(x)

        if self.use_mmd:
            y = self.FC.L0(z)
            if self.use_bn:
                y = self.FC.B0(y)
            y = self.FC.A0(y)
            if self.use_dr:
                y = self.FC.D0(y)
            return x, y
        return x
