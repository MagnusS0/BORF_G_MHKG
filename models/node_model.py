import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
from torch_geometric.utils import to_undirected, get_laplacian
from torch_sparse import SparseTensor

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
            layers.append(self.get_layer(in_features, out_features))
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()
    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        for i, layer in enumerate(self.layers):
            if self.layer_type in ["R-GCN", "R-GIN"]:
                x = layer(x, edge_index, edge_type=graph.edge_type)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        return x

class G_MHKG(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, bias=True):
        super(G_MHKG, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))
        self.filter_1 = nn.Parameter(torch.Tensor(num_nodes, 1))
        self.filter_2 = nn.Parameter(torch.Tensor(num_nodes, 1))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter_1, 0.9, 1.1)
        nn.init.uniform_(self.filter_2, 0.6, 0.8)
        nn.init.xavier_uniform_(self.W)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, eigen_vectors, lp, hp, x):
        x = torch.matmul(x, self.W)
        x_1 = torch.mm(lp, eigen_vectors.T)
        x_1 = torch.mm(x_1, x)
        x_1 = self.filter_1 * x_1
        x_1 = torch.mm(eigen_vectors, x_1)

        x_2 = torch.mm(hp, eigen_vectors.T)
        x_2 = torch.mm(x_2, x)
        x_2 = self.filter_2 * x_2
        x_2 = torch.mm(eigen_vectors, x_2)

        x = x_1 + x_2
        if self.bias is not None:
            x += self.bias
        return x

class Net(nn.Module):
    def __init__(self, num_features, nhid, num_classes, num_nodes, num_layers=2, device='cuda',
                 activation=F.relu, dropout_prob=0.3, initial_dyn_coefficient=1.1, delayed_hfd=False):
        super(Net, self).__init__()
        self.GConv1 = G_MHKG(num_features, nhid, num_nodes)
        self.layers = num_layers
        if num_layers > 2:
            self.hidden_layers = nn.ModuleList([
                G_MHKG(nhid, nhid, num_nodes)
                for _ in range(self.layers - 2)
            ])
        self.GConv2 = G_MHKG(nhid, num_classes, num_nodes)
        self.drop1 = nn.Dropout(dropout_prob)
        self.act = activation
        self.initial_dyn_coefficient = initial_dyn_coefficient
        self.delayed_hfd = delayed_hfd
        self.device = device

    def initialize_graph(self, data):
        edge_index = to_undirected(data.edge_index)
        laplacian = get_laplacian(edge_index, normalization='sym', num_nodes=data.num_nodes)
        
        # Convert Laplacian to dense tensor
        laplacian_matrix = SparseTensor(row=laplacian[0][0], col=laplacian[0][1], value=laplacian[1],
                                        sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()
        
        # Eigen decomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
        eigenvalues = eigenvalues.clamp(0, 2)
        eigenvectors = eigenvectors.to(self.device)

        # Initialize low-pass and high-pass filters based on initial_dyn_coefficient
        lp_eigen = torch.exp(self.initial_dyn_coefficient * -eigenvalues + 1)
        lp_eigen = 2 * (lp_eigen - lp_eigen.min()) / (lp_eigen.max() - lp_eigen.min())
        hp_eigen = torch.exp(eigenvalues - 1)
        hp_eigen = 2 * (hp_eigen - hp_eigen.min()) / (hp_eigen.max() - hp_eigen.min())

        # Apply delayed HFD by zeroing out first few components if specified
        if self.delayed_hfd:
            k = int(0.05 * data.num_nodes)  # Adjust k as needed, 5% of nodes for example
            lp_eigen[:k] = 0
            hp_eigen[:k] = 0

        # Convert to diagonal matrices and move to device
        self.lp = torch.diag(lp_eigen).to(self.device)
        self.hp = torch.diag(hp_eigen).to(self.device)
        self.eigen_vectors = eigenvectors

    def forward(self, data):
        x = data.x.float().to(self.device)
        x = self.GConv1(self.eigen_vectors, self.lp, self.hp, x)
        x = self.act(x)
        x = self.drop1(x)
        if self.layers > 2:
            for layer in self.hidden_layers:
                x = self.act(layer(self.eigen_vectors, self.lp, self.hp, x))
                x = self.drop1(x)
        x = self.GConv2(self.eigen_vectors, self.lp, self.hp, x)
        return F.log_softmax(x, dim=1)
