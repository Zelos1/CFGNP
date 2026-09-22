from torch import nn
from typing import Tuple
import torch

import torch
import torch.nn as nn
from typing import Tuple
from cfgnp.models import GeneralCrossAttention, GeneralSelfAttention, CFEmbedding, MoGLayer

class CFModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_nodes: int, num_obs: int, num_count: int, iterations, mog_comp):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_features = in_features
        self.num_nodes = num_nodes
        self.num_count = num_count


class IterationBlock(nn.Module):
    def __init__(self, hidden_dim: int, nodes: int, num_obs: int, in_dim: int, num_count: int):
        super().__init__()
        # store dims for checks
        self.hidden_dim = hidden_dim
        self.num_nodes = nodes
        in_dim = nodes * hidden_dim
        self.in_dim = in_dim

        
        self.l_norm1 = nn.LayerNorm((hidden_dim))
        self.l_norm2 = nn.LayerNorm((hidden_dim))
        self.sample_obs = GeneralSelfAttention(hidden_dim, 1)
        self.self_obs = GeneralSelfAttention(hidden_dim, 1) # attention among observation samples
        self.cross_obs = GeneralCrossAttention(hidden_dim, 1) # attention between observation and counterfactual embeddings
        self.self_count = GeneralSelfAttention(hidden_dim, 1) # attention among counterfactual observation samples
        self.combine_count = GeneralSelfAttention(hidden_dim, 1) # combine information among samples of counterfactual condition samples
        self.causal_est_self = GeneralSelfAttention(hidden_dim, 1) # combine observations and counterfactual information

        self.cross_obs_z = GeneralCrossAttention(hidden_dim, 1)
        self.self_z = GeneralSelfAttention(hidden_dim, 1)


    def forward(
        self,
        embedded_obs: torch.Tensor,   # shape: (B, N_obs, D, d_embed)
        embedded_z: torch.Tensor,     # shape: (B, N_count, D, d_embed)
        embedded_ast: torch.Tensor    # shape: (B, N_count, D, d_embed)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = embedded_obs.shape[0]
        attention_shape = (batch_size, -1, self.num_nodes, self.hidden_dim)
        flattened_attention_shape = (-1, self.num_nodes, self.hidden_dim)

        embedded_obs = self.sample_obs(embedded_obs.transpose(1, 2).reshape((batch_size * self.num_nodes, -1, self.hidden_dim))) \
            .reshape((batch_size, self.num_nodes, -1, self.hidden_dim)).transpose(1, 2)
        embedded_obs = nn.LeakyReLU()(embedded_obs)
        embedded_obs = self.l_norm1(embedded_obs)
        embedded_obs = self.self_obs(embedded_obs.reshape(flattened_attention_shape)).reshape(attention_shape)
        embedded_obs = nn.LeakyReLU()(embedded_obs)
        embedded_obs = self.l_norm2(embedded_obs)


        embedded_z = self.self_count(embedded_z.reshape(flattened_attention_shape)).reshape(attention_shape)
        embedded_z = nn.LeakyReLU()(embedded_z)

        concatted_count = torch.concat([embedded_z, embedded_ast], dim=-2)
        concatted_count = self.combine_count(concatted_count.reshape((-1, self.num_nodes * 2, self.hidden_dim)))\
            .reshape((batch_size, -1, self.num_nodes * 2, self.hidden_dim))
        embedded_z = concatted_count[..., :embedded_z.shape[-2], :]
        embedded_ast = concatted_count[..., -embedded_ast.shape[-2]:, :]

        embedded_z = self.cross_obs_z(embedded_obs.reshape((batch_size, -1, self.hidden_dim)), embedded_z.reshape((batch_size, -1, self.hidden_dim)))
        embedded_z = nn.SELU()(embedded_z)
        embedded_z = self.self_z(embedded_z.reshape(flattened_attention_shape)).reshape(attention_shape)
        embedded_z = nn.LeakyReLU()(embedded_z)

        # TODO maybe change the input here
        embedded_ast = self.cross_obs(embedded_z.reshape(flattened_attention_shape), embedded_ast.reshape(flattened_attention_shape)).reshape(attention_shape)

        # concatted = torch.concat([embedded_obs, embedded_z, embedded_ast], dim=1)
        # concatted = torch.unflatten(self.causal_est_self(concatted.transpose(1, 2).flatten(2)), -1, (-1, embedded_obs.shape[-1])).transpose(1, 2)
        # embedded_obs = concatted[:, :embedded_obs.shape[1]]
        # embedded_z = concatted[:, embedded_obs.shape[1]: embedded_obs.shape[1] + embedded_z.shape[1]]
        # embedded_ast = concatted[:, -embedded_ast.shape[1]:]

        return embedded_obs, embedded_z, embedded_ast


class CFNPModel(CFModel):
    def __init__(self, in_features: int, hidden_dim: int, num_nodes: int, num_obs: int, num_count: int, iterations=5, mog_comp=3):
        super().__init__(in_features, hidden_dim, num_nodes, num_obs, num_count, iterations, mog_comp)
        
        self.counterfactual_embedding = CFEmbedding(in_features, hidden_dim)
        self.obs_embedding = CFEmbedding(in_features, hidden_dim)
        self.cond_embed = CFEmbedding(in_features, hidden_dim)
        self.counterfactual_embedding_int = CFEmbedding(in_features, hidden_dim)
        self.cond_embedding_int = CFEmbedding(in_features, hidden_dim)

        self.T = iterations
        blocks = [IterationBlock(hidden_dim, num_nodes, num_obs, in_features, num_count) for _ in range(self.T)]
        self.blocks = nn.Sequential(*blocks)

        self.final_layer = MoGLayer(hidden_dim, mog_comp=mog_comp, data_dim=in_features)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        (x_ast, interventional_indices, z, obs) = x

        embedded_ast = self.counterfactual_embedding(x_ast)  # B x N_count x D x d_embed
        embedded_obs = self.obs_embedding(obs)               # B x N_obs x D x d_embed
        embedded_z = self.cond_embed(z)                      # B x N_count x D x d_embed

        embedded_ast_nonint = self.counterfactual_embedding(x_ast) # B x N_count x D x d_embed
        embedded_obs = self.obs_embedding(obs) # B x N_obs x D x d_embed
        embedded_z_nonint = self.cond_embed(z) # B x N_count x D x d_embed
        embedded_ast_int = self.counterfactual_embedding_int(x_ast.gather(2, interventional_indices.expand(-1, -1, -1, self.in_features)))
        embedded_z_int = self.counterfactual_embedding_int(z.gather(2, interventional_indices.expand(-1, -1, -1, self.in_features)))
        embedded_ast = embedded_ast_nonint.scatter(2, interventional_indices.expand(-1, -1, -1, self.hidden_dim), embedded_ast_int)
        
        embedded_z = embedded_z_nonint.scatter(2, interventional_indices.expand(-1, -1, -1, self.hidden_dim), embedded_z_int)
        # embedded_ast = embedded_ast.reshape((-1, self.hidden_dim))
        # embedded_z = embedded_z.reshape((-1, self.hidden_dim))

        embedded_ast = nn.ReLU()(embedded_ast)
        embedded_obs = nn.ReLU()(embedded_obs)
        embedded_z = nn.ReLU()(embedded_z)

        # iterate through the block list (T iterations)
        for block in self.blocks:
            embedded_obs, embedded_z, embedded_ast = block(embedded_obs, embedded_z, embedded_ast)

        embedded_ast = embedded_ast.reshape((-1, embedded_ast.shape[-1]))
        mog_output = self.final_layer(embedded_ast)
        mog_output = mog_output.reshape((x_ast.shape[0], -1, self.num_nodes, self.in_features, mog_output.shape[-2], mog_output.shape[-1]))
        # TODO add activation functions
        return mog_output

def build_cfnp_model(in_features: int, hidden_dim: int, num_obs: int, num_count: int, num_nodes: int = 5, iterations: int = 5, mog_comp: int = 3):
    return CFNPModel(in_features, hidden_dim, num_nodes, num_obs, num_count, iterations=5, mog_comp=mog_comp)






