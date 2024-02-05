""" Featurized peptide hLA dataset """

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder  # Peptide encoding
import torch
import torch.nn.functional as F
from torch import Tensor
import importlib_resources

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, AMINO_ACIDS_EXT, LOCI


class PepHLAEncoder:

    def __init__(self,
                 pep_lens: List[int],
                 aa_feature_files: List[os.PathLike] = None,
                 hla_encoding_file: os.PathLike = None,
                 allele_col_name: str = 'mhc',
                 is_pan_allele: bool = True):

        if hla_encoding_file is None:
            hla_encoding_file = importlib_resources.files('hlathena').joinpath('data').joinpath('hla_seqs_onehot.csv')

        self.hla_encoding = pd.read_csv(hla_encoding_file, index_col=allele_col_name)
        self.pep_lens = pep_lens
        self.is_pan_allele = is_pan_allele

        self.aa_feature_map = AminoAcidFeatureMap(aa_feature_files)

        pep_dim = ((self.aa_feature_map.aa_feature_count * max(self.pep_lens)) +
                                   (max(self.pep_lens) * len(AMINO_ACIDS_EXT)))
        hla_dim = len(self.hla_encoding.columns) + len(LOCI) if self.is_pan_allele else 0 # also appending number of loci categories
        pep_len_dim = len(self.pep_lens) if len(self.pep_lens) > 1 else 0
        self.feature_dimensions = pep_dim + hla_dim + pep_len_dim

    def encode_pep_len(self, pep: str):
        return F.one_hot(torch.tensor(len(pep)) % min(self.pep_lens), num_classes=len(self.pep_lens))

    def encode_peptide(self, pep):

        pep += (max(self.pep_lens) - len(pep)) * '-'

        peptide = [list(pep)]
        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS_EXT] * len(pep))
        encoder.fit(peptide)
        encoded = encoder.transform(peptide).toarray()[0]

        onehot_only = not self.aa_feature_map.aa_feature_files
        # Add additional encoding features if present
        if not onehot_only:
            aafeatmat_bd: np.ndarray = np.kron(np.eye(len(pep), dtype=int), self.aa_feature_map.aa_feature_map)
            encoded = np.concatenate((encoded, (encoded @ aafeatmat_bd)))
        return torch.as_tensor(encoded).float()

    def get_loci(self, allele):
        if 'HLA-' in allele:
            return allele[4]
        else:
            return allele[0]

    def encode_loci(self, loci):
        encoder = OneHotEncoder(categories=[LOCI])
        encoder.fit([[loci]])
        return torch.as_tensor(encoder.transform([[loci]]).toarray()[0]).float()

    def encode_hla(self, hla):
        loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        return torch.cat((loc_enc, torch.as_tensor(np.array(self.hla_encoding.loc[hla])).float()))

    def encode_pepfeats(self, pep_feats):
        return torch.as_tensor(pep_feats)

    def encode(self, pep, hla, pep_feats):
        encoding = torch.cat((self.encode_peptide(pep), self.encode_pepfeats(pep_feats)))
        if self.is_pan_allele:
            encoding = torch.cat((encoding, self.encode_hla(hla)))
        if len(self.pep_lens) > 1:
            encoding = torch.cat((encoding, self.encode_pep_len(pep)))
        return encoding

    @staticmethod
    def encode_onehot(sequences, pep_len) -> np.ndarray:
        """One hot encode peptides

        Returns:
            np.ndarray: one hot encoded peptide set
        """
        if isinstance(sequences[0], str):
            sequences = [list(s) for s in sequences]

        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS] * pep_len)
        encoder.fit(sequences)
        encoded = encoder.transform(sequences).toarray()
        return encoded

    @staticmethod
    def get_encoded_peps(sequences,
                         aafeatmat: AminoAcidFeatureMap = None) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Amino acid featurization of peptides in the class
        """

        # Split up each peptide string into individual amino acids
        if isinstance(sequences[0], str):
            sequences = [list(s) for s in sequences]

        # Input peptide sequences need to be of the same length - TO DO - handle multiple lengths
        lens = [len(s) for s in sequences]
        pep_len = lens[0]
        assert (all(l == pep_len for l in lens))  # TO DO: integrate error handling...

        onehot_encoded: np.ndarray = PepHLAEncoder.encode_onehot(sequences, pep_len)

        onehot_only = not aafeatmat

        if onehot_only:
            peps_enc = pd.DataFrame(onehot_encoded)  # , index=self.peptides)  # SISI - TO DO index on peps needed...?
        else:
            aamap = aafeatmat.aa_feature_map
            # Block diagonal aafeatmat
            aafeatmat_bd: np.ndarray = np.kron(
                np.eye(pep_len, dtype=int), aamap)
            # Feature encoding (@ matrix multiplication)
            feat_names: List[List[str]] = list(np.concatenate(
                [(f'p{i + 1}_' + aamap.columns.values).tolist()
                 for i in range(pep_len)]).flat)
            peps_enc = pd.DataFrame(onehot_encoded @ aafeatmat_bd,
                                    columns=feat_names)  # ,
            # index=self.peptides)  # SISI - TO DO index on peps needed...?

        return peps_enc

    @staticmethod
    def encode_peptides(sequences,
                         aafeatmat: AminoAcidFeatureMap = None) -> List[Tensor]:
        """Return featurized peptides

        Returns:
            List[Tensor]: featurized peptide tensors
        """
        encoded: pd.DataFrame = PepHLAEncoder.get_encoded_peps(sequences, aafeatmat)

        # TO DO: is there a more elegant way of creating tensors?...
        encoded_tensors = []
        for i in range(len(sequences)):
            encoded_peptide: Tensor = torch.as_tensor(encoded.iloc[i].values).float()
            encoded_tensors.append(encoded_peptide)
        return encoded_tensors
