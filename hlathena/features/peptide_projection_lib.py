### 
###
### Peptide projectiion funcs:
###    Peptide encoding: onehot, AA feature matrix, KF
###    Weight peptide encoding by entropy
###    Weight peptide encoding by entropy
###    PCA
###    All combined: encode_KF_wE_PCA
###

import numpy as np
import pandas as pd
#import torch
import matplotlib.pyplot as plt
import time
import os

### Peptide encoding
import sklearn.preprocessing

amino_acids = "ACDEFGHIKLMNPQRSTVWY"

def pep_encode_onehot(peps, pep_len):  
    encoder = sklearn.preprocessing.OneHotEncoder(
      categories=[list(amino_acids)] * pep_len)
    encoder.fit(peps)
    encoded = encoder.transform(peps).toarray()
    return encoded

def pep_encode_aafeatmat(peps, aafeatmat):
    
    # Input peptide sequences need to be of the same length
    pep_len = len(peps[0])
    #print(peps[0])
    assert(all(len(pep)==pep_len for pep in peps))

    # Split up each peptide string into individual amino acids
    if isinstance(peps[0], str):
        peps_split = [list(s) for s in peps]

    # One-hot (binary) encoding
    encoded = pep_encode_onehot(peps_split, pep_len)

    # Transform one-hot encoding according to AA x Feats matrix  
    # Ensure the rows have the same order as the onehot encoding 
    # This enables efficient transfprmation to other encodings 
    # by multiplication (below).
    aafeatmat = aafeatmat.loc[list(amino_acids),:]
    # Block diagonal aafeatmat
    aafeatmat_bd = np.kron(np.eye(pep_len, dtype=int), aafeatmat)
    # Feature encoding (@ matrix multiplication)
    num_feats_per_pos = aafeatmat.shape[1] # assumes AA x Feats
    feat_names = list(np.concatenate(
      [('p{0}_'.format(i+1) + aafeatmat.columns.values).tolist() for i in range(pep_len)]).flat)  
    peps_aafeatmat = pd.DataFrame(encoded @ aafeatmat_bd, columns=feat_names, index=peps)
    return peps_aafeatmat



### PCA 
def PCA_numpy_SVD(X, rowvar=False): 
    u, s, vh = np.linalg.svd(X)
    n = X.shape[0]
    sdev = s/np.sqrt(max(1, n-1))

    evals, evecs = s**2, vh.T

    explained_variances = []
    for i in range(len(evals)):
        explained_variances.append(evals[i] / np.sum(evals))
    
    return evals, evecs, explained_variances


### Weight positions
### aafeatmat assumed to be AA rows x Feats cols
def pep_pos_weight(dat, pos_weights, aafeatmat):
    num_feats_per_pos = aafeatmat.shape[1]
    peplen = int(dat.shape[1]/num_feats_per_pos)
    for i in range(peplen):
        pos_cols = 'p{0}_'.format(i+1) + aafeatmat.columns.values
        dat[pos_cols] = np.multiply(dat[pos_cols], pos_weights[i])
    return dat


### Transform
### TO DO: Assumes dat has column 'seq' that contains the peptides
def encode_KF_wE_PCA(
    tsv_file, allele, length,
    in_DIR, out_DIR, pep_col='seq', 
    use_precomp_PCA=True, dat_suffix='hits',
    plot=False):
    
    #pwd=os.path.dirname(os.path.abspath(__file__))
    print(pwd)
    
    peplen = int(peplen)
    
    pep_df = pd.read_csv(tsv_file, sep='\t')
    
    pep_df = pep_df[ (pep_df['allele']==allele) & (pep_df['length']==length)].copy()
    
    aa_code = list('GPAVLIMCFYWHKRQNEDST')
    
    sequences = pep_df['seq'].to_list()
    
    
    # Load AA feature matrix and encodfe peptides
    KFs = pd.read_csv(in_DIR + 'kideraFactors.txt', sep=' ', header=0)
    peps_KF = pep_encode_aafeatmat(dat[pep_col].values, KFs)
    
    ###  Weight positions by entropy
    molecularEntropies_MS = pd.read_csv(in_DIR + 'molecularEntropies_' + str(peplen) + '_MS.txt', sep=' ', header=0)
    molecularEntropies_IEDB = pd.read_csv(in_DIR + 'molecularEntropies_' + str(peplen) + '_IEDB.txt', sep=' ', header=0)
    molecularEntropies_MS_IEDB = (molecularEntropies_MS + molecularEntropies_IEDB)/2
    # Average of the allele-specific and pan-allele entropies so we don't miss plausible anchors/subanchors
    pos_weights = ((1-molecularEntropies_MS_IEDB.loc[allele,:]) + (1-molecularEntropies_MS_IEDB.loc['Avg',:]))/2    
    peps_KFwE = pep_pos_weight(peps_KF, pos_weights, KFs)
    peps_KFwE.shape
    
    ### PCA-transform
    if not use_precomp_PCA:
        evals, evecs, explained_variances = PCA_numpy_SVD(peps_KFwE)        
        np.savez("{0}./projection_models/PCA_KFwE_{1}_{2}.npz".format(in_DIR, allele, peplen), 
            evals=evals, evecs=evecs, explained_variances=explained_variances)
    else:
        npz_tmp = np.load("{0}./projection_models/PCA_KFwE_{1}_{2}.npz".format(in_DIR, allele, peplen))
        evecs = npz_tmp['evecs']
        explained_variances = npz_tmp['explained_variances']
    
    #print(np.sum(explained_variances).real)
    print('explained_variance per PC:')
    print(' '.join([str(np.round(ev.real,6)) for ev in explained_variances]))
    
    peps_KFwE_pca_nocenter_df = pd.DataFrame(
        np.dot(peps_KFwE, evecs), 
        index=peps_KFwE.index, 
        columns=['PC{0}'.format(i) for i in range(len(evecs))])    
    
    ### Plot
    # ref: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    if plot:
        fig, axs = plt.subplots(1,2, figsize = (15, 6))
        axs[0].plot(range(len(explained_variances)), explained_variances, 'rx-', linewidth=1)
        axs[0].set_xlabel('Principal Component')
        axs[0].set_ylabel('Proportion of Variance Explained')
        axs[1].scatter(peps_KFwE_pca_nocenter_df.iloc[:, 0], peps_KFwE_pca_nocenter_df.iloc[:, 1], 
            s=10, facecolors='none', edgecolors='black', linewidths=0.1)
        axs[1].set_title('PCA KFwE: ' + allele + ' ' + dat_suffix, fontsize=12)
        axs[1].set_xlabel('PC1', fontsize=15)
        axs[1].set_ylabel('PC2', fontsize=15)
        axs[1].axis('equal')
        #fig.show()    
        fig.savefig(out_DIR + 'KFwE_PCA_' + allele + '_' + str(peplen) + '_' + dat_suffix + '.png')
    
    return peps_KFwE_pca_nocenter_df
