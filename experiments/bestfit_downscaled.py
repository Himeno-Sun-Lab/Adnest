import os

# Overwrite values of src/default_net_params.py
net_params = {
    # 'scaling_factors_recurrent': {
    #     # Scale cortico cortical excitatory on excitatory weights
    #     'cc_scalingEtoE': 2.5,
    #     # Scale cortico cortical excitatory on inhibitory weights
    #     'cc_scalingEtoI': 2.0*2.5
    # },
    'K_scaling': 0.1,
    'fullscale_rates': os.path.join(os.getcwd(),
                                    "simulated_data",
                                    "base_theory_rates.pkl"),
    'cytoarchitecture_params': {
        'min_neurons_per_layer': 5000
    }
}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 1500.0,
    'master_seed': 2903
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 1000,
        'high': 1500
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
