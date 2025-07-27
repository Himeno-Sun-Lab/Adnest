# Overwrite values of src/default_net_params.py
import os

net_params = {
    'predictive_connectomic_params': {
        'FLN': 0.86,
        'rho_syn': 660000000.0
    },
    'cytoarchitecture_params': {
        # Path to ei ratio data
        'ei_ratio_path': os.path.join(
            os.getcwd(),
            'experimental_data',
            'fraction_EI',
            'lichtman.csv'
        ),
        # Remove layers with fewer neurons than in layer I
        'remove_smaller_layerI': False,
        # Minimal number of neurons per layer
        'min_neurons_per_layer': 5000
    },
    'connection_params': {
        'g': -2.0,
        'PSP_e': 0.1,
        'PSP_ext': 0.1,
    },
    'delay_params': {
        'distribution': 'lognormal_clipped'
    },
    'scaling_factors_recurrent': {
        'local_scaling4Eto23E': 2.0,
        'local_scalingEtoI': 1.0,
        # Scale cortico cortical excitatory on excitatory weights
        'cc_scalingEtoE': 1.75,
        # Scale cortico cortical excitatory on inhibitory weights
        'cc_scalingEtoI': 2.0*1.75
    },
    'scaling_factors_external': {
        'scaling5E': 1.05,
        'scaling6E': 1.15
    },
    'neuron_params_E': {
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
        'C_m': 220.0,
        # Membrane time constant (in ms).
        # See Allen Cells GLIF Parameters.ipynb
        # Lowered to account for high-conductance state.
        'tau_m': 10.0,
        # Time constant of postsynaptic excitatory currents (in ms).
        # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_ex': 2.0,
        # Time constant of postsynaptic inhibitory currents (in ms).
        # Value for GABA_A receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_in': 2.0,
        # Refractory period of the neurons after a spike (in ms).
        't_ref': 2.0
    },
    'neuron_params_I': {
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
        'C_m': 100.0,
        # Membrane time constant (in ms).
        # See Allen Cells GLIF Parameters.ipynb
        # Lowered to account for high-conductance state.
        'tau_m': 10.0,
        # Time constant of postsynaptic excitatory currents (in ms).
        # Value for AMPA receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_ex': 2.0,
        # Time constant of postsynaptic inhibitory currents (in ms).
        # Value for GABA_A receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_in': 2.0,
        # Refractory period of the neurons after a spike (in ms).
        't_ref': 2.0
    },
    'neuron_param_dist_E': {
        'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},
        'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},
        'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0},
    },
    'neuron_param_dist_I': {
        'V_th': {'distribution': 'lognormal', 'rel_sd': 0.0},
        'C_m': {'distribution': 'lognormal', 'rel_sd': 0.0},
        'tau_m': {'distribution': 'lognormal', 'rel_sd': 0.0},
    }
}

# Overwrite values of src/default_sim_params.py
sim_params = {
    't_sim': 12500.0,
    'V0_mean': -150.,
    'V0_sd': 50.,
}

# Parameters for the analysis
ana_params = {
    'plotRasterArea': {
        'fraction': 0.05,
        'low': 2500,
        'high': 12500
    },
    'functconn_corr': {
        'exclude_diagonal': False
    }
}
