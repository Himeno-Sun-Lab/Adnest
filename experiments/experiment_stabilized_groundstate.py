# Overwrite values of src/default_net_params.py
net_params = {}

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
