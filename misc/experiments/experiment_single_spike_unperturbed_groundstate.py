# Overwrite values of src/default_net_params.py
net_params = {}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 2500.0,
    'master_seed': 2106,
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
