import simplejson
import numpy as np
import pandas as pd
from glob import glob
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns


# layer boundaries from fig.1 in Mohan et al.
# layer V/VI divided into 1/3 & 2/3 according to von Economo & Koskinas data
layernames = ['L1', 'L2 / L3', 'L4', 'L5', 'L6']
layerborders = np.array([-252, -1201, -1582, -1979])
assert(len(layerborders) + 1 == len(layernames))

# auxiliary (broadcasted) helper variables
layerId = np.arange(len(layernames))[:, np.newaxis]
layers_bc = layerborders[:, np.newaxis]


# get all lines of all dendrites
def getDendriteLength(filename):
    # load json file
    print(filename)
    with open(filename) as fp:
        tree = simplejson.load(fp)

    # extract relevant quantities from json
    treeLines = tree['treeLines']['data']
    treePoints = np.array(tree['treePoints']['data'])
    basalDendriteTypeId = tree['customTypes']['dendrite']['id']
    apicalDendriteTypeId = tree['customTypes']['apical']['id']

    # calculate total dendritic length
    totalLength = np.zeros(len(layernames))
    for line in treeLines:
        # extract relevant properties from treeLines
        lineId, startPoint, numPoints, parentLineId, parentOffset = line
        _, parentStartPoint, parentNumPoints, _, _ = treeLines[parentLineId]

        if lineId == basalDendriteTypeId or lineId == apicalDendriteTypeId:
            # get all points, neglect radius
            linePoints = treePoints[startPoint:startPoint+numPoints, :3]
            # calculate lengths
            segLength = np.sqrt(np.sum(np.diff(linePoints, axis=0)**2, axis=1))
            # find corresponding layer for the respective piece of dendrite
            segLayerIndex = np.sum(layers_bc >= linePoints[:-1, 1], axis=0)
            # add segment lengths from corresponding layers
            totalLength += np.sum(segLength*(layerId == segLayerIndex), axis=1)

            # calculate the length of the connecting line to ParentLine
            if parentLineId == 0:
                # neglect connection to soma
                pass
            else:
                # get attachment point and first point
                parentPoint = treePoints[
                    parentStartPoint + parentNumPoints - 1 - parentOffset, :3]
                firstPoint = treePoints[startPoint, :3]
                # calculate lengths of connection to the parentLine
                segLength = np.sqrt(np.sum((parentPoint - firstPoint)**2))
                segLayerIndex = np.sum(layerborders >= parentPoint[1])
                # add the length of the connection to the parentLine
                totalLength[segLayerIndex] += segLength

    # Find the layer of the soma
    # assuming that treePoints[0] is the centre of the soma
    somaMiddleY = treePoints[0, 1]
    somaLayer = layernames[np.sum(layerborders >= somaMiddleY)]

    return totalLength, somaLayer


# Parallelize reading & processing using multiprocessing
mp_pool = mp.Pool(processes=mp.cpu_count())
mp_results = mp_pool.map(getDendriteLength, glob('morphs_atdepth_json/*.json'))

# Store results in a DataFrame
allCellsLength = pd.DataFrame(0., index=layernames, columns=layernames[1:])
for dL, sL in mp_results:
    allCellsLength[sL] += dL

# Calculate soma to cell body probability assuming a constant
# synapse density along the dendrites
s2cb = allCellsLength.div(allCellsLength.sum(axis=1), axis=0)
print(s2cb)
sns.heatmap(s2cb, vmin=0, vmax=1, cmap='Blues')
plt.title('Synapse to Cell Body Probability')
plt.xlabel('Soma')
plt.ylabel('Synapse')
plt.tight_layout()
plt.show()

# Calculate the relative number of synapses in the respective layers
# for a neuron in a given layer (also assuming constant synapse density)
relsyn = allCellsLength.div(allCellsLength.sum(axis=0), axis=1)
print(relsyn)
sns.heatmap(relsyn, vmin=0, vmax=1, cmap='Blues')
plt.xlabel('Soma')
plt.ylabel('Synapse')
plt.title('Relative Number of Synapses')
plt.tight_layout()
plt.show()
