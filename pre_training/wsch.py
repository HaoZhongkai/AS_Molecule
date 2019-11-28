#!/usr/bin/env python  
#-*- coding:utf-8 _*-

'''Schnet in a weakly supervised manner
    1. node level pre_training with node attributes Masking
    2. graph level pre_training with contrasive loss or clustering loss
    3. weakly supervised learning on specific properties
'''

import torch as th
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn
from torch.nn import Softplus
import torch.nn.init as inits
import sys

sys.path.append('..')
from base_model.schmodel import AtomEmbedding, RBFLayer, Interaction, ShiftSoftplus


class WSchnet_N(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None,
                 ):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super(WSchnet_N,self).__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.type_num = 100
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 64)
        self.atom_classifier = nn.Linear(64, 100)   # 100 denote the number of atom types
        self.ebd_dense_layer = nn.Linear(64,32)

    def set_mean_std(self, mean, std):
        self.mean_per_atom = mean.clone().detach()
        self.std_per_atom = std.clone().detach()


    def forward(self, g ):
        # g_list list of molecules


        g.edata['distance'] = g.edata['distance'].reshape(-1,1)

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        g.ndata['atom'] = atom
        # res = dgl.mean_nodes(g, "atom")
        atom = self.activation(atom)
        g.ndata["res"] = atom

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
        #
        # if self.norm:
        #     g.ndata["res"] = g.ndata[
        #         "res"] * self.std_per_atom + self.mean_per_atom

        atoms_preds = self.atom_classifier(g.ndata["res"])
        return atoms_preds


    # get whole graph embeddings by meaning the nodes
    def embed_g(self, g):
        with th.no_grad():
            # g_list list of molecules

            g.edata['distance'] = g.edata['distance'].reshape(-1, 1)

            self.embedding_layer(g)
            if self.atom_ref is not None:
                self.e0(g, "e0")
            self.rbf_layer(g)
            for idx in range(self.n_conv):
                self.conv_layers[idx](g)

            atom = self.atom_dense_layer1(g.ndata["node"])
            g.ndata['atom'] = atom
            # res = dgl.mean_nodes(g, "atom")
            # atom = self.activation(atom)
            g.ndata["res"] = atom

            if self.atom_ref is not None:
                g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
            #
            # if self.norm:
            #     g.ndata["res"] = g.ndata[
            #         "res"] * self.std_per_atom + self.mean_per_atom

            # g.ndata["res"] = self.atom_classifier(g.ndata["res"])
            embeddings_g = dgl.mean_nodes(g, 'res')
            return embeddings_g




'''G denote whole graph pre_training'''
class WSchnet_G(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 cls_dim = 2000,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None,
                 ):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super(WSchnet_G,self).__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.type_num = 100
        self.cls_dim = cls_dim
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 256)
        self.atom_dense_layer2 = nn.Linear(256,256)

        self.cls_classifier = nn.Linear(256,cls_dim)


        self.atom_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 100)
        )   # 100 denote the number of atom types
        # self.ebd_dense_layer = nn.Linear(64,32)

    def set_mean_std(self, mean, std):
        self.mean_per_atom = mean.clone().detach()
        self.std_per_atom = std.clone().detach()


    def forward(self, g ):
        # g_list list of molecules


        g.edata['distance'] = g.edata['distance'].reshape(-1,1)

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        g.ndata['atom'] = atom
        # res = dgl.mean_nodes(g, "atom")
        atom = self.activation(atom)
        atom = self.atom_dense_layer2(atom)

        g.ndata["res"] = atom

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
        #
        # if self.norm:
        #     g.ndata["res"] = g.ndata[
        #         "res"] * self.std_per_atom + self.mean_per_atom

        atoms_preds = self.atom_classifier(g.ndata["res"])
        embeddings_g = dgl.mean_nodes(g, 'res')

        cls_preds = self.cls_classifier(embeddings_g)
        return atoms_preds, cls_preds


    # get whole graph embeddings by meaning the nodes
    def embed_g(self, g):
        with th.no_grad():
            # g_list list of molecules

            g.edata['distance'] = g.edata['distance'].reshape(-1, 1)

            self.embedding_layer(g)
            if self.atom_ref is not None:
                self.e0(g, "e0")
            self.rbf_layer(g)
            for idx in range(self.n_conv):
                self.conv_layers[idx](g)

            atom = self.atom_dense_layer1(g.ndata["node"])
            # g.ndata['atom'] = atom
            # res = dgl.mean_nodes(g, "atom")
            atom = self.activation(atom)
            atom = self.atom_dense_layer2(atom)
            g.ndata["res"] = atom

            if self.atom_ref is not None:
                g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
            embeddings_g = dgl.mean_nodes(g, 'res')
            return embeddings_g



    def re_init_head(self):
        inits.xavier_normal_(self.cls_classifier.weight)
        return



'''G denote whole graph pre_training'''
class WSchnet(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """

    def __init__(self,
                 dim=64,
                 cutoff=5.0,
                 output_dim=1,
                 props_bins = 30,
                 cls_dim = 2000,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None,
                 ):
        """
        Args:
            dim: dimension of features
            output_dim: dimension of prediction
            cutoff: radius cutoff
            width: width in the RBF function
            n_conv: number of interaction layers
            atom_ref: used as the initial value of atom embeddings,
                      or set to None with random initialization
            norm: normalization
        """
        super(WSchnet,self).__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
        self.type_num = 100
        self.cls_dim = cls_dim
        self.activation = ShiftSoftplus()

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.rbf_layer = RBFLayer(0, cutoff, width)
        self.conv_layers = nn.ModuleList(
            [Interaction(self.rbf_layer._fan_out, dim) for i in range(n_conv)])

        self.atom_dense_layer1 = nn.Linear(dim, 256)
        self.atom_dense_layer2 = nn.Linear(256,256)

        # self.cls_classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Linear(64,128),
        #     nn.ReLU(),
        #     nn.Linear(128,512),
        #     nn.ReLU(),
        #     nn.Linear(512,cls_dim)
        # )
        self.cls_classifier = nn.Linear(256,cls_dim)

        self.atom_classifier = nn.Linear(256, 100)  # 100 denote the number of atom types

        self.prop_classifier = nn.Linear(256,props_bins)

        # self.ebd_dense_layer = nn.Linear(64,32)

    def set_mean_std(self, mean, std):
        self.mean_per_atom = mean.clone().detach()
        self.std_per_atom = std.clone().detach()


    def forward(self, g ):
        # g_list list of molecules


        g.edata['distance'] = g.edata['distance'].reshape(-1,1)

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        g.ndata['atom'] = atom
        # res = dgl.mean_nodes(g, "atom")
        atom = self.activation(atom)
        atom = self.atom_dense_layer2(atom)

        g.ndata["res"] = atom

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
        #
        # if self.norm:
        #     g.ndata["res"] = g.ndata[
        #         "res"] * self.std_per_atom + self.mean_per_atom

        atoms_preds = self.atom_classifier(g.ndata["res"])
        embeddings_g = dgl.mean_nodes(g, 'res')

        cls_preds = self.cls_classifier(embeddings_g)

        prop_preds = self.prop_classifier(embeddings_g)
        return atoms_preds, cls_preds, prop_preds


    # get whole graph embeddings by meaning the nodes
    def embed_g(self, g):
        with th.no_grad():
            # g_list list of molecules

            g.edata['distance'] = g.edata['distance'].reshape(-1, 1)

            self.embedding_layer(g)
            if self.atom_ref is not None:
                self.e0(g, "e0")
            self.rbf_layer(g)
            for idx in range(self.n_conv):
                self.conv_layers[idx](g)

            atom = self.atom_dense_layer1(g.ndata["node"])
            # g.ndata['atom'] = atom
            # res = dgl.mean_nodes(g, "atom")
            atom = self.activation(atom)
            atom = self.atom_dense_layer2(atom)
            g.ndata["res"] = atom

            if self.atom_ref is not None:
                g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
            embeddings_g = dgl.mean_nodes(g, 'res')
            return embeddings_g


    def inference(self,g):
        return self.embed_g(g)


    def re_init_head(self):
        inits.xavier_normal_(self.cls_classifier.weight)
        return