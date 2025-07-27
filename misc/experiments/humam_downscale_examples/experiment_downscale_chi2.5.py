# Overwrite values of src/default_net_params.py
net_params = {
    'N_scaling': 0.005,
    'K_scaling': 0.005,
    'fullscale_rates': './simulated_data/base_theory_rates.pkl',
    'scaling_type': 'linear',
    'scaling_factors_recurrent': {
        # Scale cortico cortical excitatory on excitatory weights
        'cc_scalingEtoE': 2.5,
        # Scale cortico cortical excitatory on inhibitory weights
        'cc_scalingEtoI': 2.5*2.0,
    }
}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 2500.0,
    'master_seed': 2903
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.5,
        'low': 2000,
        'high': 2500
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
