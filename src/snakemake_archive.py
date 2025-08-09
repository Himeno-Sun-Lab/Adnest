import os
import sys
import importlib.util

from helpers.snakemake import nested_dict_update, get_git_revision_hash
from default_sim_params import params as sim_params
from default_net_params import params as net_params
from network import networkDictFromDump
from simulation import Simulation

conf_name, _ = os.path.splitext(os.path.basename(sys.argv[1]))
conf_path = os.path.join(os.getcwd(), sys.argv[1])
spec = importlib.util.spec_from_file_location(conf_name, conf_path)
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

nested_dict_update(net_params, exp.net_params)
nested_dict_update(sim_params, exp.sim_params)

outpath = net_params['outpath']

# Read network hash
with open(sys.argv[2], 'r') as f:
    net_hash = f.read()

# Read simulation hash
with open(sys.argv[3], 'r') as f:
    sim_hash = f.read()

# Read analysis hash
with open(sys.argv[4], 'r') as f:
    ana_hash = f.read()

if net_params["N_scaling"] != 1.0 and net_params["K_scaling"] != 1.0:
    scale = "full"
else:
    scale = f"n_{net_params["N_scaling"]}_k_{net_params["K_scaling"]}"

if net_params["Abeta"]:
    flag = "Abeta"
    abeta = f"e_{net_params["Abeta_ratio_E"]}_i_{net_params["Abeta_raito_I"]}"
else:
    flag = "Norm"
    abeta = ""

archive_name = "_".join(scale, flag, abeta)

os.system(f"tar -czf {os.path.join(outpath, net_hash)} {os.path.join(outpath, archive_name + ".tar.gz")}" )
os.system(f"cp -r {os.path.join(outpath, net_hash, sim_hash, ana_hash, "plot")} {os.path.join(outpath, "plot", archive_name)}")

with open(sys.argv[-1], 'w') as f:
    f.write("Done")
