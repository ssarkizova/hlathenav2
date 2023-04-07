import pandas as pd 

# List of amino acid symbols
AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# Map of amino acid to numeric values
AA_MAP = pd.DataFrame(range(len(AMINO_ACIDS)), index=AMINO_ACIDS).to_dict()[0]

# Inverse map of amino acid to numeric value
INVERSE_AA_MAP = {v: k for k, v in AA_MAP.items()}