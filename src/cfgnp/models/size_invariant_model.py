import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATv2Conv, GCNConv
from cfgnp.models import ResidualGATLayer, ExponentialGATLayer, ParallelDenseGCN, CFEmbedding, MoGLayer
from cfgnp.graph_approach.causal_graph import CausalGraphFactory


class CFGNPSizeInvariant(nn.Module):
    def __init__(self, in_features, hidden_dim, num_nodes, num_obs,
                 num_count, iterations=3, mog_comp=3, num_heads_att=5, edge_fn=None, limit_paral_layers=False):
        super().__init__()

        self.hidden_dim = hidden_dim * num_heads_att
        self.in_features = in_features
        self.num_count = num_count

        self.counterfactual_embedding = CFEmbedding(in_features, self.hidden_dim)
        self.obs_embedding = CFEmbedding(in_features, self.hidden_dim)
        self.cond_embed = CFEmbedding(in_features, self.hidden_dim)

        self.counterfactual_embedding_int = CFEmbedding(in_features, self.hidden_dim)
        self.cond_embedding_int = CFEmbedding(in_features, self.hidden_dim)

        self.sequential_module = GNNAbductionSizeInvariant(
            hidden_dim, num_nodes, num_obs, num_count,
            mog_comp, num_heads_att=num_heads_att, edge_fn=edge_fn, limit_paral_layers=limit_paral_layers
        )

        self.final_layer = MoGLayer(self.hidden_dim, mog_comp=mog_comp, data_dim=in_features)

    def intervention_mask(self, int_indices, batch_size, num_nodes):
        idx = int_indices.reshape(batch_size, self.num_count, -1)

        if idx.shape[-1] == num_nodes:
            return idx.bool().unsqueeze(-1)

        # One slot wider than the graph: negative indices are routed to that scratch slot and
        # sliced off again, so they neither claim node 0 nor race a real index 0 in the same
        # sample the way a clamp would.
        mask = torch.zeros(
            (batch_size, self.num_count, num_nodes + 1),
            dtype=torch.bool,
            device=idx.device
        )
        idx = idx.long()
        mask.scatter_(2, torch.where(idx >= 0, idx, num_nodes), True)
        return mask[..., :num_nodes].unsqueeze(-1)

    def forward(self, batch):
        correct_shape = (batch.batch_size, self.num_count, -1, self.in_features)

        x_ast = batch.x_int.reshape(correct_shape)
        z = batch.x_orig.reshape(correct_shape)
        obs = batch.x_obs
        num_nodes = x_ast.shape[2]

        int_mask = self.intervention_mask(
            batch.int_indices, batch.batch_size, num_nodes
        )

        embedded_obs = self.obs_embedding(obs)

        embedded_ast = torch.where(
            int_mask,
            self.counterfactual_embedding_int(x_ast),
            self.counterfactual_embedding(x_ast)
        )

        embedded_z = torch.where(
            int_mask,
            self.cond_embedding_int(z),
            self.cond_embed(z)
        )

        embedded_ast = nn.ReLU()(embedded_ast.reshape(-1, self.hidden_dim))
        embedded_obs = nn.ReLU()(embedded_obs)
        embedded_z = nn.ReLU()(embedded_z.reshape(-1, self.hidden_dim))

        batch.x_int = embedded_ast
        batch.x_orig = embedded_z
        batch.x_obs = embedded_obs

        batch.connection_element = torch.zeros(
            (batch.batch_size, 1, num_nodes, self.hidden_dim)
        ).to(batch.x_int.device)

        batch = self.sequential_module(batch)

        embedded_ast = batch.x_int.reshape(-1, self.hidden_dim)
        mog_output = self.final_layer(embedded_ast)

        return mog_output.reshape(
            batch.batch_size,
            -1,
            num_nodes,
            self.in_features,
            mog_output.shape[-2],
            mog_output.shape[-1]
        )



class GNNAbductionSizeInvariant(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        num_obs: int,
        num_count: int,
        mog_comp=3,
        parallel_abd_layers=6,
        num_heads_att=3,
        edge_fn=None,
        limit_paral_layers=False
    ):
        super().__init__()
        self.num_obs = num_obs
        self.edge_fn = edge_fn
        if self.edge_fn is None:
            self.edge_fn = CausalGraphFactory.get_causal_graph("full").get_observation_graph_structure
        self.hidden_dim = hidden_dim * num_heads_att
        self.int_indice_dim = 3
        # TODO make this as input argument

        self.obs_gat1 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.LeakyReLU())
        self.obs_gat2 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())
        # self.obs_exp = ExponentialGATLayer(self.hidden_dim, hidden_dim, num_heads_att)
        self.obs_gat3 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())
        self.obs_gat4 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())

        self.z_gat1 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())
        self.z_exp = ExponentialGATLayer(self.hidden_dim, hidden_dim, num_heads_att)
        self.z_gat2 = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())


        self.obs_gat_second = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())
        self.z_gat_second = ResidualGATLayer(self.hidden_dim, hidden_dim, num_heads_att, nn.ReLU())


        self.conn_matching_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        max_layers = 10 if limit_paral_layers else num_nodes

        self.parallel_dmsm = ParallelDenseGCN(
            num_layers=parallel_abd_layers,
            double_num_layers=min(max_layers, num_nodes),
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim
        )

        self.local_dmsm = GCNConv(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim
        )

        self.graph_norm = torch_geometric.nn.GraphNorm(self.hidden_dim)

    def forward(self, batch):
        (embedded_obs, embedded_z, embedded_ast) = batch.x_obs, batch.x_orig, batch.x_int
        attention_shape = (batch.batch_size, self.num_obs, -1, self.hidden_dim)
        embedded_obs = embedded_obs.reshape(attention_shape)
        num_nodes = embedded_obs.shape[2]
        attention_shape = (batch.batch_size, -1, num_nodes, self.hidden_dim)
        embedded_z = embedded_z.reshape(attention_shape)
        embedded_ast = embedded_ast.reshape(attention_shape)
        

        relevant = torch.cat([
            embedded_obs,
            torch.zeros((batch.batch_size, 1, num_nodes, self.hidden_dim), device=embedded_obs.device),
        ], dim=1).reshape(batch.batch_size, -1, self.hidden_dim)


        self.obs_abduction_edges = self.edge_fn(num_nodes, self.num_obs)
        new_edges, z_edges = self.compute_edges(batch, embedded_obs, relevant, num_nodes)

        # 1. Abduction (OBS)
        x = relevant.reshape(-1, self.hidden_dim)

        # Learn information from observations
        x = self.obs_gat1(x, new_edges, batch.batch_size)
        x = self.obs_gat2(x, new_edges, batch.batch_size)
        x = self.obs_gat3(x, new_edges, batch.batch_size)
        x = self.obs_gat4(x, new_edges, batch.batch_size)

        obs_reshaped = x.reshape(attention_shape)

        embedded_obs = obs_reshaped[:, :embedded_obs.shape[1]]
        conn = obs_reshaped[:, embedded_obs.shape[1]:]

        z_input = torch.cat([embedded_z, conn], dim=1)
        z_input = z_input.reshape(batch.batch_size, -1, self.hidden_dim)

        # Combine z information. This however should probably not use z_edges / the conn element from obs
        z = self.z_gat1(z_input.reshape(-1, self.hidden_dim), z_edges, batch.batch_size)
        z = self.z_gat2(z, z_edges, batch.batch_size)

        z = z.reshape(attention_shape)

        embedded_z = z[:, :embedded_z.shape[1]]
        conn = z[:, embedded_z.shape[1]:]

        z_input = torch.cat([embedded_z, conn], dim=1)
        z = self.z_gat_second(z_input.reshape(-1, self.hidden_dim), z_edges, batch.batch_size)
        z = z.reshape(attention_shape)

        embedded_z = z[:, :embedded_z.shape[1]]
        conn = z[:, embedded_z.shape[1]:]

        # 2. Action

        concatted = torch.cat([
            conn,
            embedded_ast
        ], dim=2).reshape(-1, self.hidden_dim)

        gat_out = self.parallel_dmsm(concatted, batch.edge_index)
        gat_out += F.relu(self.local_dmsm(concatted, batch.edge_index))
        gat_out = gat_out / 2

        gat_out = self.graph_norm(gat_out, batch.batch)

        concatted = gat_out.reshape(batch.batch_size, -1, 2 * num_nodes, self.hidden_dim)
        embedded_ast = concatted[:, :, -num_nodes:]

        batch.x_obs = embedded_obs.reshape(batch.x_obs.shape)
        batch.x_orig = embedded_z.reshape(batch.x_orig.shape)
        batch.x_int = embedded_ast.reshape(batch.x_int.shape)

        return batch

    def compute_edges(self, batch, embedded_obs, relevant, num_nodes):
        """
        New edges: Edges that combine observation graphs into one small graph for each element of the batch
        z_edges: Connect exogenous nodes with z for each batch element
        """
        
        device = embedded_obs.device

        base_edges = self.obs_abduction_edges.to(device)

        num_edges = base_edges.size(1)
        nodes_per_relevant_graph = relevant.size(1)

        offsets = (
            torch.arange(batch.batch_size, device=device)
            .repeat_interleave(num_edges)
            * nodes_per_relevant_graph
        )

        new_edges = base_edges.repeat(1, batch.batch_size) + offsets.unsqueeze(0)

        numbered_nodes = torch.arange(num_nodes * 2, device=device)
        z_edges = torch.stack([numbered_nodes % num_nodes, numbered_nodes])
        z_edges = z_edges.repeat(1, batch.batch_size)

        z_offsets = (
            torch.arange(batch.batch_size, device=device)
            .repeat_interleave(num_nodes * 2)
            * num_nodes * 2
        )

        z_edges = z_edges + z_offsets.unsqueeze(0)

        return new_edges.long(), z_edges.long()

def build_cfnp_gnn_model(in_features: int, hidden_dim: int, num_obs: int, num_count: int, num_nodes: int, iterations: int = 5, mog_comp: int = 3, num_heads_att=3, limit_paral_layers=False):
    model = CFGNPSizeInvariant(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        num_obs=num_obs,
        num_count=num_count,
        iterations=iterations,
        mog_comp=mog_comp,
        num_heads_att=num_heads_att,
        limit_paral_layers=limit_paral_layers
    )
    return model