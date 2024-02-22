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
import re

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, AMINO_ACIDS_EXT, LOCI, AA_MAPPING, BOS_TOKEN, BOS_DICT


class PepHLAEncoder:  # TODO: add support for no hla encoding, if none, just return pep + pep feats tensor


    def __init__(self,
                 pep_lens: List[int],
                 # maxlen: int = 11,
                 # pep_length: int,
                 aa_feature_files: List[os.PathLike] = None,
                 hla_encoding_file: os.PathLike = None,
                 allele_col_name: str = 'mhc',
                 is_pan_allele: bool = True):

        if hla_encoding_file is None:
            hla_encoding_file = importlib_resources.files('hlathena').joinpath('data').joinpath('hla_seqs_onehot.csv')
        self.hla_encoding = pd.read_csv(hla_encoding_file, index_col=allele_col_name)

        ### Adding for transformer testing
        hla_seq_file = importlib_resources.files('hlathena').joinpath('data').joinpath( 'ABCG_prot.parsed.clean.SEQ.ME.ALL.FEATS.txt')
        self.hla_seqs = pd.read_csv(hla_seq_file, sep=' ', index_col='allele')

        self.hla_mapping = {hla: i+22 for i, hla in enumerate(self.hla_encoding.index)} # FOR V1
        # self.pep_len = pep_length
        # self.maxlen = maxlen
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

    def encode_peptide_notflat_with_BOS(self, pep):

        pep = self.add_padding(pep, max(self.pep_lens), padding_token='-')

        padding_mask = self.get_padding_mask(pep)
        # padding_bool = [aa != "-" for aa in pep]
        # TODO: need this unsqueeze part?
        # padding_mask = torch.tensor(padding_bool).unsqueeze(1)

                # 'B' + pep + ((max(self.pep_lens) - len(pep)) * '-'))
        # pep += (max(self.pep_lens) - len(pep)) * '-'

        peptide = [list(pep)]
        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS_EXT] * len(pep))
        encoder.fit(peptide)
        encoded = encoder.transform(peptide).toarray()[0].reshape(len(pep),len(AMINO_ACIDS_EXT))

        # onehot_only = not self.aa_feature_map.aa_feature_files
        # # Add additional encoding features if present
        # if not onehot_only:
        #     aafeatmat_bd: np.ndarray = np.kron(np.eye(len(pep), dtype=int), self.aa_feature_map.aa_feature_map)
        #     encoded = np.concatenate((encoded, (encoded @ aafeatmat_bd)))
        return torch.as_tensor(encoded).float(), padding_mask

    def enumerate_pep(self, pep):
        padded_seq = pep.ljust(max(self.pep_lens), '-')
        return torch.tensor([AA_MAPPING[aa] for aa in padded_seq])

    def enumerate_pHLA(self, pep, hla):
        # first_last_four = seq[:4] + seq[-4:]
        first_last_four_enumerated = torch.tensor([AA_MAPPING[aa] for aa in pep[:4]+pep[-4:]])
        hla_enumerated = torch.as_tensor([self.hla_mapping[hla]])
        return torch.cat((first_last_four_enumerated, hla_enumerated))

    def BOS_tensor(self):
        return torch.tensor([BOS_DICT[BOS_TOKEN]])

    def get_loci(self, allele):
        if 'HLA-' in allele:
            return allele[4]
        else:
            return allele[0]

    def encode_loci(self, loci):
        encoder = OneHotEncoder(categories=[LOCI])
        encoder.fit([[loci]])
        return torch.as_tensor(encoder.transform([[loci]]).toarray()[0]).float()

    def add_padding(self, seq, max_len, padding_token="-"):
        """
        Pads the amino acid sequence to a maximum length by adding the necessary number of padding tokens ("-") after the fourth amino acid

        Parameters:
        - sequence (str): input amino acid sequence for peptide
        - max_len (int): maximum length you want to pad the sequence to
        - padding_token (str): token used for padding

        Returns:
        - str: padded amino acid sequence
        """
        difference = max_len - len(seq)
        final_seq = 'B' + seq[:4] + padding_token * difference + seq[4:]
        return final_seq

    def get_padding_mask(self, seq):
        padding_bool = [aa != "-" for aa in seq]
        # TODO: need this unsqueeze part?
        return torch.tensor(padding_bool).unsqueeze(1)

    # for transformer v2 implementation
    def encode_hla_fullseq_notflat(self, hla):
        if 'HLA-' in hla: # cleaning name for testing file
            hla = re.sub(r'[*:]', '', hla.split('HLA-')[1])
        # loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        hla_seq = [list(self.hla_seqs.at[hla, 'seq'])]
        padding_mask = self.get_padding_mask(hla_seq[0])

        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS_EXT] * len(hla_seq[0]))
        encoder.fit(hla_seq)
        encoded = encoder.transform(hla_seq).toarray()[0].reshape(len(hla_seq[0]), len(AMINO_ACIDS_EXT))

        return torch.as_tensor(encoded).float(), padding_mask

    def encode_hla_fullseq(self, hla):
        if 'HLA-' in hla: # cleaning name for testing file
            hla = re.sub(r'[*:]', '', hla.split('HLA-')[1])
        # loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        hla_seq = [list(self.hla_seqs.at[hla, 'seq'])]
        encoder = OneHotEncoder(
            categories=[AMINO_ACIDS_EXT] * len(hla_seq[0]))
        encoder.fit(hla_seq)
        encoded = encoder.transform(hla_seq).toarray()[0]#.reshape(len(hla_seq[0]), len(AMINO_ACIDS_EXT))

        return torch.as_tensor(encoded).float()

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
