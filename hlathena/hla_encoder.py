### TO Do - pull out code that encodes the hla alleles (e.g. given a list of alleles from the peptide_dataset df)
### this is where we introduce tensors

from sklearn import preprocessing

import torch

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, INVERSE_AA_MAP


class HLAEncoder:
    def __init__(
        self):

        # Check all same length
        # Encode or static method for encode
        self.something = None

    def encode_hla(hla : str) ->  torch.Tensor:
        enc = torch.zeros(20*len(hla), dtype=torch.int8)
        return enc