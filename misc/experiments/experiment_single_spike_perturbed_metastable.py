# Overwrite values of src/default_net_params.py
net_params = {
    'scaling_factors_recurrent': {
        # Scale cortico cortical excitatory on excitatory weights
        'cc_scalingEtoE': 2.5,
        # Scale cortico cortical excitatory on inhibitory weights
        'cc_scalingEtoI': 5.0
    },
    'single_spike': {('pericalcarine', 'IV', 'E'): 2000.}
}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 2500.0,
    'master_seed': 2106
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 1500,
        'high': 1600
    },
    'functconn_corr': {
        'exclude_diagonal': False
    },
    'python_sort': False
}
