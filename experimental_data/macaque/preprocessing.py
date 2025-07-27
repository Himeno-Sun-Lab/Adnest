import numpy as np
import pandas as pd


replace_this = ['PERIRHINAL', '8L', '8B', 'CORE', 'ENTORHINAL', 'INSULA',
                'TEMPORAL_POLE', 'V3A']
replace_for = ['PERI', '8l', '8b', 'Core', 'ENTO', 'INS', 'POLE', 'V3a']

# Import density data from Beul & Hilgetag NeuroImage 2019
density_data = pd.read_excel('density_data.xlsx').dropna()
density_data = density_data.replace(replace_this, replace_for)
# Compute density ratio
dens_source = density_data['density'].values[np.newaxis, :]
dens_target = density_data['density'].values[:, np.newaxis]
density_ratio = pd.DataFrame(
    data=dens_target/dens_source, index=density_data['area'],
    columns=density_data['area']).unstack()
density_ratio.index.rename('SOURCE', level=0, inplace=True)
density_ratio.index.rename('TARGET', level=1, inplace=True)
density_ratio = density_ratio.reset_index()
density_ratio.rename(columns={0: 'DENS_RATIO'}, inplace=True)
density_ratio['DENS_LOGRATIO'] = np.log(density_ratio['DENS_RATIO'].values)

# Import FLN data from Markov et al. PNAS 2013
fln_data = pd.read_excel('PNAS_2013.xls')
fln_data = fln_data.replace(replace_this, replace_for)
fln_data = fln_data.drop(columns=['STATUS', 'BIBLIOGRAPHY'])

# Import SLN data from Chaudhuri et al. Neuron 2015
sln_data = pd.read_excel('Neuron_2015_Table.xlsx')
sln_data = sln_data.replace(replace_this, replace_for)
sln_data = sln_data.drop(columns='FLN')

# Merge data & export
merged_data = fln_data.merge(
    sln_data, how='left', on=['SOURCE', 'TARGET']).merge(
    density_ratio, how='left', on=['SOURCE', 'TARGET']).dropna()
# Exclude projections with FLN value lower than 10e-5
merged_data = merged_data.loc[merged_data['FLNe'] >= 10e-5]
merged_data.to_pickle('macaque_data_merged.pkl')
