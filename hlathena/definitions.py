import pandas as pd 

# List of supported peptide lengths
PEP_LENS = [8, 9, 10, 11, 12]

# List of amino acid symbols
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

AMINO_ACIDS_EXT = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '.', '-']

## Map of amino acid to numeric values
AA_MAP = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))

## Inverse map of amino acid to numeric value
INVERSE_AA_MAP = {v: k for k, v in AA_MAP.items()}