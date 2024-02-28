import pandas as pd 
import importlib_resources


# List of supported peptide lengths
PEP_LENS = [8, 9, 10, 11, 12]

# List of amino acid symbols
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

AMINO_ACIDS_EXT = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-', 'B']

LOCI = ['A','B','C','G']

AA_MAPPING = { #B = BOS token
    'B': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
    'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9,
    'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14,
    'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19,
    'V': 20, '-': 21
}

BOS_TOKEN = "<BOS>"
BOS_DICT = {"<BOS>": 0}

# AMINO_ACIDS_EXT = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.', '-']

## Map of amino acid to numeric values
AA_MAP = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

## Inverse map of amino acid to numeric value
INVERSE_AA_MAP = {v: k for k, v in AA_MAP.items()}

# Built-in amino acid feature files
aa_feature_file_PCA3 = importlib_resources.files('hlathena').joinpath('data').joinpath('aafeatmat_AAPropsPCA3.txt')
aa_feature_file_Kidera = importlib_resources.files('hlathena').joinpath('data').joinpath('aafeatmat_KideraFactors.txt')
aa_feature_file_PCA3 = importlib_resources.files('hlathena').joinpath('data').joinpath('aasimmat_BLOSUM62.txt')
aa_feature_file_PCA3 = importlib_resources.files('hlathena').joinpath('data').joinpath('aasimmat_PMBEC.txt')
