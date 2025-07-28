import os

# Overwrite values of src/default_net_params.py
net_params = {}

net_params["N_scaling"] = 0.1
net_params["K_scaling"] = 0.1
net_params["fullscale_rates"] = os.path.join(os.getcwd(), "simulated_data", "base_theory_rates.pkl")

net_params["Abeta"] = True
net_params["Abeta_ratio_E"] = 0.6
net_params["Abeta_ratio_I"] = 0.2
net_params["neuron_model_Abeta_E"] = "iaf_psc_alpha"
net_params["neuron_model_Abeta_I"] = "iaf_psc_alpha"

net_params['neuron_params_Abeta_E'] = {
    # Leak potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'E_L': -65.0,
    # Threshold potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_th': -50.0,
    # Membrane potential after a spike (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_reset': -70.0,
    # Membrane capacitance (in pF).
    # See Allen Cells GLIF Parameters.ipynb
    'C_m': 280.0,
    # Membrane time constant (in ms).
    # See Allen Cells GLIF Parameters.ipynb
    # Lowered to account for high-conductance state.
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
    'tau_syn_ex': 2.0,
    # Time constant of postsynaptic inhibitory currents (in ms).
    # Set as the same value as tau_syn_ex
    'tau_syn_in': 2.0,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0,
}
net_params["neuron_params_Abeta_I"] = {
    # Leak potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'E_L': -70.0,
    # Threshold potential of the neurons (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_th': -45.0,
    # Membrane potential after a spike (in mV).
    # See Allen Cells GLIF Parameters.ipynb
    'V_reset': -70.0,
    # Membrane capacitance (in pF).
    # See Allen Cells GLIF Parameters.ipynb
    'C_m': 120.0,
    # Membrane time constant (in ms).
    # See Allen Cells GLIF Parameters.ipynb
    # Lowered to account for high-conductance state.
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
    'tau_syn_ex': 2.0,
    # Time constant of postsynaptic inhibitory currents (in ms).
    # Set as the same value as tau_syn_ex
    'tau_syn_in': 2.0,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0,
}

net_params['neuron_param_dist_Abeta_E'] = {
    'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},  # 0.21
    'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},   # 0.22
    'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0}, # 0.55
}
net_params['neuron_param_dist_Abeta_I'] = {
    'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},  # 0.22
    'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},   # 0.34
    'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0}, # 0.43
}
# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 4500.0,
    'master_seed': 2903
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 3500,
        'high': 4000
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
