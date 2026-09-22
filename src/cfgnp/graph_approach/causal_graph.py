import numpy as np
import torch

class CausalGraph:
    def __init__(self):
        self.name = "full"

    def single_graph(self, num_nodes):
        raise NotImplementedError

    def dual_graph(self, num_nodes):
        single_graph_structure = self.single_graph(num_nodes)
        dual = single_graph_structure + num_nodes
        
        dual_connections = np.stack([
            np.concat([np.arange(num_nodes), np.arange(num_nodes) + num_nodes]),
            np.concat([np.arange(num_nodes) + num_nodes, np.arange(num_nodes)])
        ], axis=0)
    
        dmsm_structure = torch.from_numpy(
            np.concatenate([dual, dual_connections], axis=1)
        )
        return dmsm_structure

    def get_observation_graph_structure(self, num_nodes, num_obs):
        """
        This returns the graph structure of the num_obs observations but the edges of the exogenous variables are reversed. 
        The edges are only for one batch. All edges go to one additional fresh nodeset.
        """

        single_graph = self.single_graph(num_nodes)

        obs_graph = single_graph.repeat(num_obs, axis=0).reshape(2, -1)
        # Disconnected obs graphs. Obs graphs are fully connected
        all_obs = obs_graph + np.arange(num_obs).repeat(single_graph.shape[1]) * num_nodes

        # The connecting indices from a node in an observation to the matching node in the fresh node batch q. 
        # q combines the information of the observations. q does not have any edges
        connecting_indices = all_obs.max() + 1 + np.arange(num_nodes).reshape((1, -1)).repeat(num_obs, axis=0).flatten()
        exogenous_connection = np.stack([np.arange(all_obs.max() + 1), connecting_indices])

        entire_graph = np.concat([all_obs, exogenous_connection], axis=1)
        return torch.from_numpy(entire_graph)


class FullCausalGraph(CausalGraph):
    def __init__(self):
        self.name = "full"
        
    def single_graph(self, num_nodes):
        single_node_set = np.arange(num_nodes)
        
        single_graph_structure = np.stack([
            single_node_set.repeat(num_nodes),
            single_node_set.reshape((1, -1))
            .repeat(num_nodes, axis=0)
            .flatten()
        ], axis=0)
        return single_graph_structure


class CausalGraphFactory:
    def get_causal_graph(name) -> CausalGraph:
        if name == "full":
            return FullCausalGraph()