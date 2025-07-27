import os
import numpy as np
from copy import deepcopy


blueprint = '''# Overwrite values of src/default_net_params.py
import os

net_params = {
    'predictive_connectomic_params': {
        'FLN': {FLN},
        'rho_syn': {rho_syn}
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
        'g': {g},
        'PSP_e': 0.1,
        'PSP_ext': 0.1,
    },
    'delay_params': {
        'distribution': '{delay_distribution}'
    },
    'scaling_factors_recurrent': {
        'local_scaling4Eto23E': 2.0,
        'local_scalingEtoI': 1.0,
        # Scale cortico cortical excitatory on excitatory weights
        'cc_scalingEtoE': {cc_scalingEtoE},
        # Scale cortico cortical excitatory on inhibitory weights
        'cc_scalingEtoI': 2.0*{cc_scalingEtoE}
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
        'tau_syn_ex': {tau_syn_ex},
        # Time constant of postsynaptic inhibitory currents (in ms).
        # Value for GABA_A receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_in': {tau_syn_in},
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
        'tau_syn_ex': {tau_syn_ex},
        # Time constant of postsynaptic inhibitory currents (in ms).
        # Value for GABA_A receptors from (Fourcaud & Brunel, 2002)
        'tau_syn_in': {tau_syn_in},
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
'''


rho_syn = 6.6e8
delay_distribution = 'lognormal_clipped'
FLN = .86
tau_syn_ex = 2.
tau_syn_in = 4.
g = -5.
CC_SCALING = [1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3., 4.]
for cc_scalingEtoE in CC_SCALING:
    cc_scalingEtoI = cc_scalingEtoE
    experiment = deepcopy(blueprint)
    new_dict = {
            '{FLN}': str(FLN),
            '{g}': str(g),
            '{rho_syn}': str(rho_syn),
            '{delay_distribution}': delay_distribution,
            '{tau_syn_ex}': str(tau_syn_ex),
            '{tau_syn_in}': str(tau_syn_in),
            '{cc_scalingEtoE}': str(cc_scalingEtoE),
            '{cc_scalingEtoI}': str(cc_scalingEtoI),
            }

    for key, val in new_dict.items():
        experiment = experiment.replace(key, val)

    fn = os.path.join('experiments', 'exp_fraction_EI')
    for key, val in new_dict.items():
        fn += '_' + key[1:-1] + '_' + val
    fn += '.py'

    if os.path.isfile(fn):
        print(f'File exists, will not overwrite {fn}')
    else:
        with open(fn, 'w') as f:
            f.write(experiment)
