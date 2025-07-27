# Overwrite values of src/default_net_params.py
net_params = {
    'scaling_factors_recurrent': {
        # Scale cortico cortical excitatory on excitatory weights
        'cc_scalingEtoE': 2.5,
        # Scale cortico cortical excitatory on inhibitory weights
        'cc_scalingEtoI': 2.0*2.5
    }
}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 12500.0,
    'master_seed': 2903
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 12000,
        'high': 12500
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
