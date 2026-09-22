import os
import numpy as np
import torch


def generate_rcm_dataset(num_samples, num_obs, seed, regions_per_node, dir="./data/rcm_data"):
    """
    Generate a Regional Canonical Model (RCM) dataset.
    
    Regional Canonical Model (RCM): An SCM <V, U_RCM, F_RCM, P(U_RCM)> where:
    1. U_RCM = {RC : ∀C ∈ C(G)} where C(G) is the set of maximally confounded cliques
       and RC ~ Unif(0,1) for all RC ∈ U_RCM.
    2. For each V ∈ V, denote R_V = {RC : ∀C s.t. V ∈ C}. Each V is associated with
       a set of regions I_V = {I_V,i} where each I_V,i = {[a,b]_C,i : RC ∈ R_V}.
       Each I_V,i is associated with a function h_V,i: D_PaV → D_V.
    3. Each f_RCM_V is defined as:
       f_RCM_V(paV, rV) = h_V,i*(paV) where i* is the largest i s.t.
       rC ∈ [a,b]_C,i for all rC ∈ rV, else h_V,DEFAULT(paV).
    
    Args:
        num_samples: Number of samples to generate
        num_obs: Number of observational samples per intervention
        seed: Random seed
        dir: Directory to save dataset
        regions_per_node: Number of regions per node (complexity hyperparameter r)
    
    Returns:
        Dataset object with samples
    """
    if os.path.exists(f'./data/rcm_data/dataset_{seed}.pt'):
        samples = torch.load(f'{dir}/dataset_{seed}.pt')
        return samples
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    all_nodes = ['Z', 'X', 'M', 'Y']
    
    U_RCM = {f"RC_{node}": np.random.uniform(0, 1, size=num_samples) for node in all_nodes}
    cond_observations = create_rcm_observations(num_samples, all_nodes, regions_per_node, U_RCM)
    count_noises = {f"RC_{node}": np.random.normal(loc=U_RCM[f'RC_{node}'], scale=1, size=num_samples) for node in all_nodes}
    int_observations = create_rcm_observations(num_samples, all_nodes, regions_per_node, count_noises)
    U_RCM_obs = {f"RC_{node}": np.random.uniform(0, 1, size=num_samples * num_obs) for node in all_nodes}
    observations = create_rcm_observations(num_samples * num_obs, all_nodes, regions_per_node, U_RCM_obs)

    samples = []
    #TODO include indexes here?
    for i in range(num_samples):
        intervention_indices = np.array([1])
        sample_int = np.array(list(int_observations[i].values()))[intervention_indices]
        zeros = np.zeros((4))
        zeros[intervention_indices] = sample_int
        sample_int = zeros
        int_obs = np.array(list(cond_observations[i].values()))
        cond_obs = np.array(list(cond_observations[i].values()))
        obs = np.array([list(observations[j].values()) for j in range(i * num_obs, (i + 1) * num_obs)])
        samples.append(((torch.tensor(sample_int).unsqueeze(0), torch.tensor(intervention_indices).unsqueeze(0), torch.tensor(cond_obs).unsqueeze(0),
                         torch.tensor(obs)), torch.tensor(int_obs).unsqueeze(0)))
    
    os.makedirs(dir, exist_ok=True)
    torch.save(samples, f'{dir}/dataset_{seed}.pt')

    return samples
    

def create_rcm_observations(num_samples, all_nodes, regions_per_node, U_RCM):
    
    # Define the hardcoded graph structure
    # NODES: M, Z, X, Y
    # EDGES: Z -> X, Z -> M, Z -> Y, X -> Y, X -> M, M -> Y
    graph = {
        'Z': [],
        'X': ['Z'],
        'M': ['Z', 'X'],
        'Y': ['Z', 'X', 'M']
    }

    # Define intervals for each node
    # Each node V gets regions_per_node intervals within [0,1] for each RC in R_V
    node_intervals = {}
    for node in all_nodes:
        node_intervals[node] = []
        for i in range(regions_per_node):
            a = i / regions_per_node
            b = (i + 1) / regions_per_node
            node_intervals[node].append((a, b))
    
    # Define structural equations based on the causal graph
    def h_default(pa_values):
        """Default function h_V,DEFAULT when no region matches"""
        if isinstance(pa_values, dict) and pa_values:
            return float(np.mean(list(pa_values.values())))
        else:
            return 0.5
    
    def find_region_index(rc_value, intervals):
        """Find largest i* such that rc_value ∈ [a,b]_i"""
        for i in range(len(intervals) - 1, -1, -1):  # largest i first
            a, b = intervals[i]
            if a <= rc_value < b:
                return i
        return -1  # no match
    
    def create_structural_equation(node, parents):
        """Create f_RCM_V for a given node"""
        def f_rcm(pa_values, rc_value, sample_idx):
            """f_RCM_V(paV, rV) - evaluate structural equation for node"""
            # Find region i*: largest i such that rc_value ∈ [a,b]_i
            i_star = find_region_index(rc_value, node_intervals[node])
            
            if i_star >= 0:
                # Apply region-specific function h_V,i*
                # Simple implementation: weighted average of parents with region contribution
                if parents:
                    parent_sum = sum(pa_values.values())
                    parent_avg = parent_sum / len(parents)
                    region_contribution = i_star / regions_per_node
                    return float(0.7 * parent_avg + 0.3 * region_contribution)
                else:
                    # Root node
                    return float(i_star / regions_per_node)
            else:
                # No matching region, use default
                return h_default(pa_values)
        
        return f_rcm
    
    # Create structural equations for all nodes
    structural_eqs = {node: create_structural_equation(node, graph.get(node, []))
                      for node in all_nodes}
    
    # Generate samples using topological order
    samples = []
    for sample_idx in range(num_samples):
        sample = {}
        for node in all_nodes:
            parents = graph.get(node, [])
            pa_values = {parent: sample[parent] for parent in parents if parent in sample}
            rc_key = f"RC_{node}"
            rc_value = U_RCM[rc_key][sample_idx]
            sample[node] = structural_eqs[node](pa_values, rc_value, sample_idx)
        samples.append(sample)
    
    return samples    


# quick smoke test
if __name__ == "__main__":
    ds = generate_rcm_dataset(num_samples=10000, num_obs=100, seed=42, dir="./data/rcm_test",
                              regions_per_node=20)
    print(ds)
    print("len", len(ds))
    inp, targ = ds[0]
    print("intervened shape:", inp[0].shape)
    print("original shape  :", inp[1].shape)
    print("obs block shape :", inp[2].shape)
    print("target shape    :", targ.shape)
