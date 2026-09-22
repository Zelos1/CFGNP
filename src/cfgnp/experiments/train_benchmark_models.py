from benchmarking.causal_flow_matching.generic_causal_cfm import triangle_cfm_model, loan_cfm_model, mshape_cfm_model
from benchmarking.ot_bcm.generic_causal_bgm import loan_bgm_model, mshape_bgm_model, triangle_bgm_model
from cfgnp.train_suite import train_run

def train_all_benchmark_mshape(seed):
    mshape_cfm_model(seed=seed)
    mshape_bgm_model(seed=seed)

def train_all_benchmark_triangle(seed):
    triangle_bgm_model(seed=seed)
    triangle_cfm_model(seed=seed)

def train_all_benchmark_loan(seed):
    loan_bgm_model(seed=seed)
    loan_cfm_model(seed=seed)