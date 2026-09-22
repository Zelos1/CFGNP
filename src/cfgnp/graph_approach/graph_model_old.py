from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.nn import Sequential as GSequential
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import numpy as np

from cfgnp.models import CFEmbedding, GeneralSelfAttention, GeneralCrossAttention, MoGLayer, CFModel
from cfgnp.util.util import get_observation_graph_structure


class GNNIterationNP(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, num_obs: int, num_count: int, mog_comp=3, parallel_abd_layers=3, num_heads_att=3):
        super().__init__()
        in_features = num_nodes * hidden_dim
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim * num_heads_att
        self.l_norm1 = nn.LayerNorm((self.hidden_dim))
        self.l_norm4 = nn.LayerNorm((self.hidden_dim))

        self.obs_abduction_edges = get_observation_graph_structure(num_nodes, num_obs)

        self.self_obs = GeneralSelfAttention(self.hidden_dim, num_heads_att) # attention among observation samples
        self.sample_obs = GeneralSelfAttention(self.hidden_dim, num_heads_att)
        self.self_obs_2 = GeneralSelfAttention(in_features * num_heads_att, num_heads_att)
        self.cross_obs = GeneralCrossAttention(in_features * num_heads_att, num_heads_att) # attention between observation and counterfactual embeddings
        self.causal_est_self = GeneralSelfAttention(num_count + num_obs, 1) # combine observations and counterfactual information
        self.cross_obs_z = GeneralCrossAttention(self.hidden_dim, num_heads_att)
        self.self_z = GeneralSelfAttention(self.hidden_dim, num_heads_att)
        self.parallel_abd_layers = parallel_abd_layers

        self.int_indice_dim = 3

        self.int_indice_layer = nn.Sequential(*[
            nn.Linear(self.num_nodes, self.int_indice_dim * self.num_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.int_indice_dim * num_nodes, self.int_indice_dim *  num_nodes),
            nn.LeakyReLU(),
            nn.Linear(self.int_indice_dim * num_nodes, self.int_indice_dim * num_nodes),
            nn.ReLU()
        ])

        self.conn_matching_layer = nn.Linear(self.hidden_dim, self.hidden_dim + self.int_indice_dim)

        self.l_norm2 = nn.LayerNorm((self.hidden_dim))
        self.l_norm3 = nn.LayerNorm((self.hidden_dim))
        self.merging_factor = nn.Parameter(torch.tensor(0.5))
        self.merging_factor_2 = nn.Parameter(torch.tensor(0.5))
        self.merging_factor_3 = nn.Parameter(torch.tensor(0.5))

        self.gat_obs = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)
        self.gat_obs_2 = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)
        self.gat_z = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)
        self.gat_z_2 = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)

        # self.dense_dmsm = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.dense_dmsm_list = nn.ModuleList()
        for _ in range(parallel_abd_layers):
            layer_list = []
            for _ in range(num_nodes // 2):
                layer_list.append((GCNConv(in_channels=self.hidden_dim + self.int_indice_dim, out_channels=self.hidden_dim + self.int_indice_dim), "x, edge_index -> x"))
                layer_list.append(nn.LeakyReLU(0.1))
                layer_list.append(nn.Dropout(0.3))
            layer_list.append((GCNConv(in_channels=self.hidden_dim + self.int_indice_dim, out_channels=self.hidden_dim), "x, edge_index -> x"))
            layer_list.append(nn.LeakyReLU(0.1))
            layer_list.append(nn.Dropout(0.3))
            dense_dmsm = GSequential("x, edge_index", layer_list)
            self.dense_dmsm_list.append(dense_dmsm)
        self.gat_dmsm = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)
        self.gat_dmsm_2 = GATv2Conv(in_channels=self.hidden_dim, out_channels=hidden_dim, heads=num_heads_att, concat=True, dropout=0.1)
        self.graph_norm = torch_geometric.nn.GraphNorm(self.hidden_dim)
        self.obs_graph_norm = torch_geometric.nn.GraphNorm(self.hidden_dim)
        # self.graph_norm_2 = torch_geometric.nn.GraphNorm(hidden_dim)
    
    def forward(self, batch):
        # embedded_z: B x N_count x D x d_embed
        (embedded_obs, embedded_z, embedded_ast) = batch.x_obs, batch.x_orig, batch.x_int
        attention_shape = (batch.batch_size, -1, self.num_nodes, self.hidden_dim)
        flatten_attention_shape = (-1, self.num_nodes, self.hidden_dim)
        embedded_obs_reshaped = embedded_obs.reshape(attention_shape)
        embedded_z_reshaped = embedded_z.reshape(attention_shape)
        embedded_ast_reshaped = embedded_ast.reshape(attention_shape)
        int_indices = batch.int_indices.reshape((batch.batch_size, 1, -1)).expand(-1, embedded_ast_reshaped.shape[1], -1)
        int_indices_onehot = F.one_hot(int_indices, self.num_nodes)
        int_indices_onehot = int_indices_onehot.sum(dim=-2).to(dtype=embedded_ast_reshaped.dtype)
        int_indices_onehot = self.int_indice_layer(int_indices_onehot)

        embedded_obs_copy = embedded_ast_reshaped

        relevant_things = torch.concat([embedded_obs_reshaped, batch.connection_element, embedded_z_reshaped], axis=1).reshape((batch.batch_size, -1, self.hidden_dim))
        add_indices = torch.arange(batch.batch_size).repeat(self.obs_abduction_edges.shape[1]) * relevant_things.shape[1]
        new_edges = self.obs_abduction_edges.repeat(1, batch.batch_size)
        new_edges += add_indices

        obs_infos = self.gat_obs(relevant_things.reshape((-1, self.hidden_dim)).to(int_indices_onehot.device), edge_index=new_edges.to(int_indices_onehot.device))
        obs_infos = nn.LeakyReLU()(obs_infos)
        obs_infos = self.obs_graph_norm(obs_infos)
        obs_infos = self.gat_obs_2(obs_infos, edge_index=new_edges.to(int_indices_onehot.device))
        obs_infos = nn.ReLU()(obs_infos)

        obs_infos_reshaped = obs_infos.reshape(attention_shape)
        embedded_obs_reshaped = obs_infos_reshaped[:, :embedded_obs_reshaped.shape[1], :, :]
        conn_element = obs_infos_reshaped[:, embedded_obs_reshaped.shape[1]:-embedded_z_reshaped.shape[1], :, :]

        numbered_nodes = torch.arange(self.num_nodes * 2)
        z_edges = torch.stack([numbered_nodes % self.num_nodes, numbered_nodes]).repeat(1, batch.batch_size).to(device=int_indices_onehot.device)

        offsets = torch.arange(batch.batch_size, device=int_indices_onehot.device) * self.num_nodes * 2
        offsets = offsets.repeat_interleave(self.num_nodes * 2)
        z_edges = z_edges + offsets

        relevant_things = torch.concat([embedded_z_reshaped, batch.connection_element], axis=1).reshape((batch.batch_size, -1, self.hidden_dim))
        z_infos = self.gat_z(relevant_things.reshape((-1, self.hidden_dim)), z_edges)
        z_infos = nn.ReLU()(z_infos)
        z_infos = self.l_norm1(z_infos)
        z_infos = self.gat_z_2(z_infos, z_edges).reshape(attention_shape)

        embedded_z_reshaped = z_infos[:, :embedded_z_reshaped.shape[1], :, :]
        embedded_z = embedded_z_reshaped.reshape((-1, self.hidden_dim))
        conn_element = z_infos[:, embedded_z_reshaped.shape[1]:, :, :]
        conn_element = nn.ReLU()(conn_element)

        # # 1. Abduction step
        # embedded_obs_reshaped = self.sample_obs(embedded_obs_reshaped.transpose(1, 2).reshape((batch.batch_size * self.num_nodes, -1, self.hidden_dim))) \
        #     .reshape((batch.batch_size, self.num_nodes, -1, self.hidden_dim)).transpose(1, 2)
        # embedded_obs_reshaped = nn.LeakyReLU()(embedded_obs_reshaped)
        # embedded_obs_reshaped = self.self_obs(embedded_obs_reshaped.reshape(flatten_attention_shape)).reshape(attention_shape)
        # embedded_obs_reshaped = nn.LeakyReLU()(embedded_obs_reshaped)
        # embedded_obs_reshaped = self.l_norm1(embedded_obs_reshaped)
        # embedded_obs_reshaped = embedded_obs_reshaped + self.merging_factor_3 * embedded_obs_copy

        # embedded_ast_reshaped_copy = embedded_ast_reshaped

        # # Use GNN to specifically learn exogenous variables from the observational dataset. 
        # # obs should also be used to learn abduction for exogenous variables. -> use cross attention from obs to z to learn exogenous variables.
        # # Use the result as input to the graph attention with the interventional structure.
        # z_cross_infos = self.cross_obs_z(embedded_obs_reshaped.reshape((batch.batch_size, -1, self.hidden_dim)), embedded_z_reshaped.reshape((batch.batch_size, -1, self.hidden_dim)))
        # z_cross_infos = nn.SELU()(z_cross_infos.reshape(attention_shape))
        # z_own_infos = self.self_z(embedded_z_reshaped.reshape(flatten_attention_shape)).reshape(attention_shape)
        # z_own_infos = nn.LeakyReLU()(embedded_z_reshaped)
        # embedded_z_reshaped = z_cross_infos + z_own_infos
        # # rather add the attention outputs than to chain them

        # Not sure if this is needed, because exogenous variables should be enough to learn the graph values.
        # embedded_ast_reshaped = self.cross_obs(embedded_obs_reshaped.flatten(2), embedded_ast_reshaped.flatten(2)).reshape(attention_shape)
        # embedded_ast_reshaped = nn.SELU()(embedded_ast_reshaped)
        # embedded_ast_reshaped = self.l_norm2(embedded_ast_reshaped)
        # embedded_ast_reshaped = embedded_ast_reshaped + embedded_ast_reshaped_copy * self.merging_factor_2

        embedded_ast_positional = torch.concat([embedded_ast_reshaped, int_indices_onehot.reshape(batch.batch_size, -1, self.num_nodes, self.int_indice_dim)], dim=-1)

        concatted_count = torch.concat([self.conn_matching_layer(conn_element), embedded_ast_positional], dim=2).reshape((-1, self.hidden_dim + self.int_indice_dim))

        # embedded_ast = torch.reshape(embedded_ast_reshaped * int_indices_onehot.unsqueeze(-1), embedded_ast.shape)
        # concatted_count = torch.concat([conn_element, embedded_ast.reshape(attention_shape)], dim=2).reshape((-1, self.hidden_dim))

        # TODO maybe first learn the source nodes. That might impose some constraints on the graph which might help learn.
        # also, this should be easier than trying to learn the entire scm with function.

        # 2. Action

        # gat_result = self.gat_dmsm(x=concatted_count, edge_index=batch.edge_index.to(device=embedded_obs.device))
        # gat_result = nn.LeakyReLU(0.1)(gat_result)

        gat_result = torch.zeros((concatted_count.shape[0], self.hidden_dim), device=embedded_obs.device)
        for l in self.dense_dmsm_list:
            gat_result += l(x=concatted_count.to(embedded_obs.device), edge_index=batch.edge_index.to(device=embedded_obs.device))
        gat_result = gat_result / self.parallel_abd_layers
        gat_result = self.graph_norm(gat_result, batch.batch.to(device=embedded_obs.device))
        
        concatted_count = gat_result.reshape((batch.batch_size, -1, 2 * self.num_nodes, self.hidden_dim))
        # embedded_z_reshaped = concatted_count[:, :, :self.num_nodes]
        embedded_ast_reshaped = concatted_count[:, :, -self.num_nodes:]

        
        # TODO what about conections in graph depending on intervened nodes. -> Not sure if this is possible without knowing the underlying graph
        # TODO check if it works better to only have one/two iterations but have the same layer after each other multiple times.
        

        # TODO what about cooperative message passing?
        # Generally, the interventional nodes should only be broadcasting and not listening. However, the other nodes still have impact in the DMSM
        # model, hence they have to be chosen in a way that the intervention stays the same. Therefore, int nodes should still listen
        # The others should broadcast and listen.
        # Except for one node. In the graph (because DAG) there must be at least one node with in_degree 0. This node should only listen. This node
        # is the target node in general. However, for varying target nodes, this might influence training


        batch.x_obs = embedded_obs_reshaped.reshape(embedded_obs.shape)
        batch.x_orig = embedded_z_reshaped.reshape(embedded_z.shape)
        batch.x_int = embedded_ast_reshaped.reshape(embedded_ast.shape)
        batch.connection_element = conn_element

        return batch

class GNNCFNPModel(CFModel):
    def __init__(self, in_features: int, hidden_dim: int, num_nodes: int, num_obs: int, num_count, iterations=3, mog_comp=3, num_heads_att=3):
        super().__init__(in_features, hidden_dim * num_heads_att, num_nodes, num_obs, num_count, iterations, mog_comp)
        self.counterfactual_embedding = CFEmbedding(in_features, self.hidden_dim)
        self.obs_embedding = CFEmbedding(in_features, self.hidden_dim)
        self.cond_embed = CFEmbedding(in_features, self.hidden_dim)
        self.counterfactual_embedding_int = CFEmbedding(in_features, self.hidden_dim)
        self.cond_embedding_int = CFEmbedding(in_features, self.hidden_dim)

        self.iterative_module = [GNNIterationNP(hidden_dim, num_nodes, num_obs, num_count, mog_comp, num_heads_att=num_heads_att) for _ in range(iterations)]
        self.iterative_module = nn.Sequential(*self.iterative_module)
        
        self.T = iterations
        self.final_layer = MoGLayer(self.hidden_dim, mog_comp=mog_comp, data_dim=in_features)

    def forward(self, batch) -> torch.Tensor:
        """
        Receives x_ast as the 'onehot-encoded' sample that contains the intervened indices.
        
        :param self: 
        :param x_ast: interventioned indice with value
        :type x_ast: torch.Tensor
        :param z: previous observation conditional
        :type z: torch.Tensor
        :param obs: observations
        :type obs: torch.Tensor
        :return: mixture of Gaussians for each node
        :rtype: Tensor
        """
        interventional_indices = batch.int_indices.reshape((batch.batch_size, self.num_count, -1))
        interventional_indices = interventional_indices.unsqueeze(-1)
        correct_shape = (batch.batch_size, self.num_count, -1, self.in_features)
        (x_ast, z, obs) = (batch.x_int, batch.x_orig, batch.x_obs)
        x_ast = x_ast.reshape(correct_shape)
        z = z.reshape(correct_shape)
        # N_count should usually be 1
        embedded_ast_nonint = self.counterfactual_embedding(x_ast) # B x N_count x D x d_embed
        embedded_obs = self.obs_embedding(obs) # B x N_obs x D x d_embed
        embedded_z_nonint = self.cond_embed(z) # B x N_count x D x d_embed
        embedded_ast_int = self.counterfactual_embedding_int(x_ast.gather(2, interventional_indices.expand(-1, -1, -1, self.in_features)))
        embedded_z_int = self.counterfactual_embedding_int(z.gather(2, interventional_indices.expand(-1, -1, -1, self.in_features)))
        embedded_ast = embedded_ast_nonint
        embedded_ast.scatter_(2, interventional_indices.expand(-1, -1, -1, self.hidden_dim), embedded_ast_int)
        
        embedded_z = embedded_z_nonint
        embedded_z = embedded_z.scatter(2, interventional_indices.expand(-1, -1, -1, self.hidden_dim), embedded_z_int)
        embedded_ast = embedded_ast.reshape((-1, self.hidden_dim))
        embedded_z = embedded_z.reshape((-1, self.hidden_dim))

        embedded_ast = nn.ReLU()(embedded_ast)
        embedded_obs = nn.ReLU()(embedded_obs)
        embedded_z = nn.ReLU()(embedded_z)
        batch.x_int = embedded_ast
        batch.x_orig = embedded_z
        batch.x_obs = embedded_obs

        # Includes 1. Abduction and 2. Action
        batch.connection_element = torch.zeros((batch.batch_size, 1, self.num_nodes, self.hidden_dim)).to(batch.x_int.device)
        batch = self.iterative_module(batch)
        embedded_obs = batch.x_obs
        embedded_z = batch.x_orig
        embedded_ast = batch.x_int

        # 3. Prediction
        embedded_ast = embedded_ast.reshape((-1, self.hidden_dim)) # Should this be in the shape? Maybe each node should have own layer.
        mog_output = self.final_layer(embedded_ast)
        mog_output = mog_output.reshape((batch.batch_size, -1, self.num_nodes, self.in_features, mog_output.shape[-2], mog_output.shape[-1]))
        return mog_output


def build_cfnp_gnn_model(in_features: int, hidden_dim: int, num_obs: int, num_count: int, num_nodes: int, iterations: int = 5, mog_comp: int = 3, num_heads_att=3) -> GNNCFNPModel:
    model = GNNCFNPModel(
        in_features=in_features,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        num_obs=num_obs,
        num_count=num_count,
        iterations=iterations,
        mog_comp=mog_comp,
        num_heads_att=num_heads_att
    )
    return model