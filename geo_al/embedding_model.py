import torch as th
import torch.nn as nn
import numpy as np
import dgl
import dgl.function as fn
from torch.nn import Softplus


#cannot first write device in model
class AtomEmbedding(nn.Module):
    """
    Convert the atom(node) list to atom embeddings.
    The atom with the same element share the same initial embeddding.
    """
    def __init__(self, dim=128, type_num=100, pre_train=None):
        """
        Randomly init the element embeddings.
        Args:
            dim: the dim of embeddings
            type_num: the largest atomic number of atoms in the dataset
            pre_train: the pre_trained embeddings
        """
        super(AtomEmbedding, self).__init__()
        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, g, p_name="node"):
        """Input type is dgl graph"""
        atom_list = g.ndata["nodes"]
        g.ndata[p_name] = self.embedding(atom_list)
        return g.ndata[p_name]


class EdgeEmbedding(nn.Module):
    """
    Convert the edge to embedding.
    The edge links same pair of atoms share the same initial embedding.
    """
    def __init__(self, dim=128, edge_num=3000, pre_train=None):
        """
        Randomly init the edge embeddings.
        Args:
            dim: the dim of embeddings
            edge_num: the maximum type of edges
            pre_train: the pre_trained embeddings
        """
        super(EdgeEmbedding, self).__init__()
        self._dim = dim
        self._edge_num = edge_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train,
                                                          padding_idx=0)
        else:
            self.embedding = nn.Embedding(edge_num, dim, padding_idx=0)

    def generate_edge_type(self, edges):
        """
        Generate the edge type based on the src&dst atom type of the edge.
        Note that C-O and O-C are the same edge type.
        To map a pair of nodes to one number, we use an unordered pairing function here
        See more detail in this disscussion:
        https://math.stackexchange.com/questions/23503/create-unique-number-from-2-numbers
        Note that, the edge_num should larger than the square of maximum atomic number
        in the dataset.
        """
        atom_type_x = edges.src["node_type"]
        atom_type_y = edges.dst["node_type"]

        return {
            "type":
            atom_type_x * atom_type_y +
            (th.abs(atom_type_x - atom_type_y) - 1)**2 / 4
        }

    def forward(self, g, p_name="edge_f"):
        g.apply_edges(self.generate_edge_type)
        g.edata[p_name] = self.embedding(g.edata["type"])
        return g.edata[p_name]


class ShiftSoftplus(Softplus):
    """
    Shiftsoft plus activation function:
        1/beta * (log(1 + exp**(beta * x)) - log(shift))
    """
    def __init__(self, beta=1, shift=2, threshold=20):
        super().__init__(beta, threshold)
        self.shift = shift
        self.softplus = Softplus(beta, threshold)

    def forward(self, input):
        return self.softplus(input) - np.log(float(self.shift))


class RBFLayer(nn.Module):
    """
    Radial basis functions Layer.
    e(d) = exp(- gamma * ||d - mu_k||^2)
    default settings:
        gamma = 10
        0 <= mu_k <= 30 for k=1~300
    """
    def __init__(self, low=0, high=30, gap=0.1, dim=1):
        super(RBFLayer, self).__init__()
        self._low = low
        self._high = high
        self._gap = gap
        self._dim = dim

        self._n_centers = int(np.ceil((high - low) / gap))
        centers = np.linspace(low, high, self._n_centers)
        self.centers = nn.Parameter(th.Tensor(centers), requires_grad=False)
        self._fan_out = self._dim * self._n_centers

        self._gap = centers[1] - centers[0]

    def dis2rbf(self, edges):
        dist = edges.data["distance"]
        radial = dist - self.centers
        coef = float(-1 / self._gap)
        rbf = th.exp(coef * (radial**2))
        return {"rbf": rbf}

    def forward(self, g):
        """Convert distance scalar to rbf vector"""
        g.apply_edges(self.dis2rbf)
        return g.edata["rbf"]


class CFConv(nn.Module):
    """
    The continuous-filter convolution layer in SchNet.
    One CFConv contains one rbf layer and three linear layer
        (two of them have activation funct).
    """
    def __init__(self, rbf_dim, dim=64, act="sp"):
        """
        Args:
            rbf_dim: the dimsion of the RBF layer
            dim: the dimension of linear layers
            act: activation function (default shifted softplus)
        """
        super(CFConv, self).__init__()
        self._rbf_dim = rbf_dim
        self._dim = dim

        self.linear_layer1 = nn.Linear(self._rbf_dim, self._dim)
        self.linear_layer2 = nn.Linear(self._dim, self._dim)

        if act == "sp":
            self.activation = nn.Softplus(beta=0.5, threshold=14)
        else:
            self.activation = act

    def update_edge(self, edges):
        rbf = edges.data["rbf"]
        h = self.linear_layer1(rbf)
        h = self.activation(h)
        h = self.linear_layer2(h)
        return {"h": h}

    def forward(self, g):
        g.apply_edges(self.update_edge)
        g.update_all(message_func=fn.u_mul_e('new_node', 'h', 'neighbor_info'),
                     reduce_func=fn.sum('neighbor_info', 'new_node'))
        return g.ndata["new_node"]


class Interaction(nn.Module):
    """
    The interaction layer in the SchNet model.
    """
    def __init__(self, rbf_dim, dim):
        super(Interaction, self).__init__()
        self._node_dim = dim
        self.activation = nn.Softplus(beta=0.5, threshold=14)
        self.node_layer1 = nn.Linear(dim, dim, bias=False)
        self.cfconv = CFConv(rbf_dim, dim, act=self.activation)
        self.node_layer2 = nn.Linear(dim, dim)
        self.node_layer3 = nn.Linear(dim, dim)

    def forward(self, g):

        g.ndata["new_node"] = self.node_layer1(g.ndata["node"])
        cf_node = self.cfconv(g)
        cf_node_1 = self.node_layer2(cf_node)
        cf_node_1a = self.activation(cf_node_1)
        new_node = self.node_layer3(cf_node_1a)
        g.ndata["node"] = g.ndata["node"] + new_node
        return g.ndata["node"]


class SchEmbedding(nn.Module):
    """
    SchNet Model from:
        Schütt, Kristof, et al.
        SchNet: A continuous-filter convolutional neural network
        for modeling quantum interactions. (NIPS'2017)
    """
    def __init__(
        self,
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
        super(SchEmbedding, self).__init__()
        self.name = "SchNet"
        self._dim = dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm
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
        self.atom_dense_layer2 = nn.Linear(64, output_dim)

    def set_mean_std(self, mean, std):
        self.mean_per_atom = mean.clone().detach()
        self.std_per_atom = std.clone().detach()

    def forward(self, g):
        # g_list list of molecules

        # g = dgl.batch([mol.ful_g for mol in mol_list])
        g.edata['distance'] = g.edata['distance'].reshape(-1, 1)
        # g.to(device)

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        g.ndata['atom'] = atom
        res = dgl.mean_nodes(g, "atom")
        atom = self.activation(atom)
        g.ndata["res"] = atom

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        if self.norm:
            g.ndata["res"] = g.ndata[
                "res"] * self.std_per_atom + self.mean_per_atom

        preds = self.atom_dense_layer2(dgl.mean_nodes(g, "res"))
        return preds

    def inference(self, g):
        # g_list list of molecules

        # g = dgl.batch([mol.ful_g for mol in mol_list])
        g.edata['distance'] = g.edata['distance'].reshape(-1, 1)
        # g.to(device)

        self.embedding_layer(g)
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)
        for idx in range(self.n_conv):
            self.conv_layers[idx](g)

        atom = self.atom_dense_layer1(g.ndata["node"])
        g.ndata['atom'] = atom
        res = dgl.mean_nodes(g, "atom")
        # atom = self.activation(atom)
        # g.ndata["res"] = atom
        #
        # if self.atom_ref is not None:
        #     g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]
        #
        # if self.norm:
        #     g.ndata["res"] = g.ndata[
        #                          "res"] * self.std_per_atom + self.mean_per_atom

        # preds = self.atom_dense_layer2(dgl.mean_nodes(g, "res"))
        return res
