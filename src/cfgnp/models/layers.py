import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn import Sequential as GSequential
from torch.nn import Module as m_c

class ResidualGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads, activation, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.activation = activation
        self.norm = torch_geometric.nn.GraphNorm(out_dim * heads)

    def forward(self, x, edge_index, batch_size):
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(x.size(0) // batch_size)
        x_res = x
        x = self.gat(x, edge_index)
        x = self.activation(x)
        x = self.norm(x, batch)
        return x + x_res


class ExponentialGATLayer(nn.Module):
    """
    GAT -> LogSoftmax -> concat with input -> linear combine
    """
    def __init__(self, in_dim, out_dim, heads, dropout=0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * out_dim * heads, out_dim * heads),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        x_new = self.gat(x, edge_index)
        x_new = torch.log_softmax(x_new, dim=-1)
        return self.combine(torch.cat([x_new, x], dim=-1))


class ParallelDenseGCN(nn.Module):
    def __init__(self, num_layers, double_num_layers, in_dim, hidden_dim, graph_layer=None):
        super().__init__()

        graph_layer = GATv2Conv if graph_layer is None else graph_layer
        # graph_layer = GCNConv if graph_layer is None else graph_layer
        

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            layer_list = []
            for _ in range(double_num_layers // 2):
                layer_list.append((graph_layer(in_dim, in_dim), "x, edge_index -> x"))
                layer_list.append(nn.LeakyReLU(0.1))
                layer_list.append(nn.Dropout(0.3))

            layer_list.append((graph_layer(in_dim, hidden_dim), "x, edge_index -> x"))
            layer_list.append(nn.ReLU(0.1))

            self.layers.append(GSequential("x, edge_index", layer_list))

        self.combine = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x, edge_index):
        outs = [layer(x=x, edge_index=edge_index) for layer in self.layers]
        return self.combine(torch.cat(outs, dim=-1))
    

class CFEmbedding(nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if in_features < out_features:
            if in_features <= out_features // 2:
                self.embedding = nn.Linear(in_features, out_features)
            else:
                self.embedding = nn.Sequential(
                    nn.Linear(in_features, out_features // 2),
                    nn.ReLU(),
                    nn.Linear(out_features // 2, out_features),
                    nn.ReLU()
                )
        else:
            self.embedding = nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.ReLU(),
                nn.Linear(in_features // 2, out_features),
                nn.ReLU()
            )

    def forward(self, x):
        return self.embedding(x)

class GeneralCrossAttention(m_c):
    def __init__(self, in_dim: int, num_heads: int):
        super().__init__()
        # self.match_dim = nn.Linear(in_dim, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True, dropout=0.1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Docstring for forward
        
        :param self: 
        :param x: Input that is used as key and value -> determines output shape
        :param y: Input that is used as query
        """
        # x = self.match_dim(x)
        # x = nn.ReLU()(x)
        x, _ = self.attention(y, x, x)
        return x

class GeneralSelfAttention(m_c):
    def __init__(self, in_dim:int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.match_dim = nn.Linear(in_dim, embed_dim)
        
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads, batch_first=True, dropout=0.1)
    
    def forward(self, x):
        # x = self.match_dim(x)
        # x = nn.ReLU()(x)
        x, _ = self.attention(x, x, x)
        return x

class MoGLayer(m_c):
    def __init__(self, input_features, mog_comp, data_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dim = data_dim
        self.mog_comp = mog_comp
        self.first_linear = nn.Linear(input_features, input_features)
        self.mean_lin = nn.Linear(input_features, mog_comp * data_dim)
        self.var_lin = nn.Linear(input_features, mog_comp * data_dim)
        self.weight_mog = nn.Linear(input_features, mog_comp * data_dim)
    
    def forward(self, x):
        x = self.first_linear(x)
        x = nn.ReLU()(x)
        mean = torch.unflatten(self.mean_lin(x), -1, (self.data_dim, self.mog_comp))
        var = torch.unflatten(self.var_lin(x), -1, (self.data_dim, self.mog_comp))
        var = nn.Softplus()(var)
        weight_mog = torch.unflatten(self.weight_mog(x), -1, (self.data_dim, self.mog_comp))
        weight_mog = nn.Softmax(dim=-1)(weight_mog)
        return torch.stack([mean, var, weight_mog], dim=-1)