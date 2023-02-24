import logomaker
from collections import Counter
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

def plot_length(tsv_file, allele):
    
    ncol_plot = 1
    fig, axs = plt.subplots(1, ncol_plot, sharex=False, sharey=False, figsize=(4.5*ncol_plot, 4));
    pep_df = pd.read_csv(tsv_file, sep='\t')
    pep_df = pep_df[pep_df['allele']==allele].copy()
    ax = axs
    pep_df['length'].value_counts().sort_index().plot.bar(ax=ax);
    ax.set_title('Length distribution {} (n={})'.format(allele, pep_df.shape[0]));

def plot_logo(tsv_file, allele, length):
    
    pep_df = pd.read_csv(tsv_file, sep='\t')
    
    pep_df = pep_df[ (pep_df['allele']==allele) & (pep_df['length']==length)].copy()
    
    aa_code = list('GPAVLIMCFYWHKRQNEDST')
    
    sequences = pep_df['seq'].to_list()
    
    aa_counts = pd.Series([pd.DataFrame(sequences)[0].str.slice(i,i+1).str.cat() for i in range(0,length)]).apply(Counter)
    aa_freq = pd.concat([pd.DataFrame(aa_counts[i],columns=aa_code,index=[i]) for i in range(0,length)]).fillna(0)
    aa_freq_norm = aa_freq.div(aa_freq.sum(axis=1),axis=0)
    R = np.log2(20) - stats.entropy(aa_freq_norm, base=2, axis=1)
    logo_df = aa_freq_norm.mul(R, axis=0)
    
    logo = logomaker.Logo(df = logo_df)
    logo.ax.set_title('Logo plot {} (n={})'.format(allele, logo_df.shape[0]));
    