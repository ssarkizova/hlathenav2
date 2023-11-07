# TO Do - pull out code that encodes the peptides (e.g. given a list of peptides form the peptide_dataset df)
# this is where we introduce tensors

import pandas as pd
import numpy as np 
from typing import List

from sklearn import preprocessing

import torch
from torch import Tensor

import os
from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, INVERSE_AA_MAP


class PepEncoder:
    
    # TO DO ?
    def encode_pep(pep: str) -> torch.Tensor:
        enc = torch.zeros(20*len(pep), dtype=torch.int8)
        return enc

    # TO DO ? could we use a multi-dim tensor?? decoding tricky
    # @staticmethod
    def encode_peps(peps) -> List[torch.Tensor]:
        enc = []
        return enc
        
    @staticmethod
    def encode_onehot(sequences, pep_len) -> np.ndarray:
        """One hot encode peptides

        Returns:
            np.ndarray: one hot encoded peptide set
        """       
        if isinstance(sequences[0], str):
            sequences = [list(s) for s in sequences]
        
        encoder = preprocessing.OneHotEncoder(
            categories=[AMINO_ACIDS] * pep_len)
        encoder.fit(sequences)
        encoded = encoder.transform(sequences).toarray()
        return encoded
    
    
    @staticmethod
    def get_encoded_peps(sequences,
                         aafeatmat: AminoAcidFeatureMap=None) -> pd.DataFrame:
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
        assert(all(l==pep_len for l in lens)) # TO DO: integrate error handling...

        onehot_encoded: np.ndarray = PepEncoder.encode_onehot(sequences, pep_len)

        onehot_only = not aafeatmat

        if onehot_only:
            peps_enc = pd.DataFrame(onehot_encoded) #, index=self.peptides)  # SISI - TO DO index on peps needed...?
        else:
            aamap = aafeatmat.aa_feature_map
            # Block diagonal aafeatmat
            aafeatmat_bd: np.ndarray = np.kron(
                np.eye(pep_len, dtype=int), aamap)
            # Feature encoding (@ matrix multiplication)
            feat_names: List[List[str]] = list(np.concatenate(
                [(f'p{i+1}_' + aamap.columns.values).tolist()
                 for i in range(pep_len)]).flat)
            peps_enc = pd.DataFrame(onehot_encoded @ aafeatmat_bd,
                                    columns=feat_names) #,
                                    #index=self.peptides)  # SISI - TO DO index on peps needed...?

        return peps_enc


    # TO DO - needs to be edited from where this code was some vars are not avalable here
    def decode_peptide(self, encoded: Tensor) -> List[str]:
        """Decode peptide tensor and return peptide

        Args:
            encoded (Tensor): encoded peptide

        Returns:
            str: decoded peptide
        """
        
        #if self.peptide_features_dim > 0:
        #    encoded = encoded[:-self.peptide_features_dim]

        encoded = encoded.reshape(
            #self.peptide_length, self.aa_feature_map.feature_count)
            -1, len(AMINO_ACIDS)) # TO DO: dims for anything other than onehot?
        dense = encoded.argmax(-1)
        if len(dense.shape) > 1:
            peptide = [''.join([INVERSE_AA_MAP[aa.item()]
                               for aa in p]) for p in dense]
        else:
            peptide = ''.join([INVERSE_AA_MAP[aa.item()] for aa in dense])
        return peptide


    # TO DO
    def _encode_peptides(sequences,
                         aafeatmat: AminoAcidFeatureMap=None) -> List[Tensor]:
        """Return featurized peptides
        
        Returns:
            List[Tensor]: featurized peptide tensors
        """
        encoded: pd.DataFrame = PepEncoder.get_encoded_peps(sequences, aafeatmat)
        
        # TO DO: is there a more elegant way of creating tensors?...
        encoded_tensors = []
        for i in range(len(sequences)):
            encoded_peptide: Tensor = torch.as_tensor(encoded.iloc[i].values).float()
            encoded_tensors.append(encoded_peptide)
        return encoded_tensors