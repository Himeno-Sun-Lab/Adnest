# Human Neuron Morphology Analysis

Analysis based on fully reconstructed human neuron morphologies from Christiaan deKock's lab available through the HBP collab: https://collab.humanbrainproject.eu/#/collab/528/nav/4669

The data is described in Mohan et al. [Cereb. Cortex 2015](https://academic.oup.com/cercor/article/25/12/4839/311644)

1. Download the data.
2. Convert the `.ASC` files to `.json`: Open the `.ASC` file with the [HBP Morphology Viewer](https://neuroinformatics.nl/HBP/morphology-viewer/) and save it as `.json` into the folder `morphs_json`.
3. Run `placeSomaAtDepth.py` to place the morphologies at the correct cortical depth.
4. Run `dendriteLength.py` for the analysis.
