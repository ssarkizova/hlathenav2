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
import esm

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

        # self.hla_mapping = {hla: i+22 for i, hla in enumerate(self.hla_encoding.index)} # FOR V1
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

        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()

    def get_hla_seq(self, hla_name):
        if 'HLA-' in hla_name: # cleaning name for testing file
            hla_name = re.sub(r'[*:]', '', hla_name.split('HLA-')[1])
        # loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        # hla_seq = [list(self.hla_seqs.at[hla, 'seq'])]
        return self.hla_seqs.at[hla_name, 'seq']
    
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

        pep = self.add_padding_and_BOS(pep, max(self.pep_lens), padding_token='-')

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

    def enumerate_pep_with_BOS(self, pep):
        padded_seq = self.add_padding_and_BOS(pep, max(self.pep_lens))
        padding_mask = self.get_padding_mask(padded_seq)
        return torch.tensor([AA_MAPPING[aa] for aa in padded_seq]), padding_mask

    def enumerate_hla_with_BOS(self, hla):
        if 'HLA-' in hla: # cleaning name for testing file
            hla = re.sub(r'[*:]', '', hla.split('HLA-')[1])
        # loc_enc = self.encode_loci(self.get_loci(hla)) # encoding allele loci
        hla_seq = 'B' + self.hla_seqs.at[hla, 'seq']
        padding_mask = self.get_padding_mask(hla_seq)
        return torch.tensor([AA_MAPPING[aa] for aa in hla_seq]), padding_mask

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

    def add_padding(self, sequence, max_len, padding_token = "-"):
        """
        Pads the amino acid sequence to a maximum length by adding the necessary number of padding tokens ("-") after the fourth amino acid
    
        Parameters: 
        - sequence (str): input amino acid sequence for peptide
        - max_len (int): maximum length you want to pad the sequence to 
        - padding_token (str): token used for padding
        
        Returns: 
        - str: padded amino acid sequence
        """
        difference = max_len - len(sequence)
        return sequence[:4] + padding_token*difference + sequence[4:]

    def add_padding_and_BOS(self, seq, max_len, padding_token="-"):
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
        return torch.tensor(padding_bool).unsqueeze(0)

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


    def transform_peptide_hla_concat(
        self,
        peptide_seq, 
        hla, 
        esm_model = None, 
        esm_batch_converter = None, 
        esm_alphabet = None, 
        encode_choice = "embed"
    ):

        if esm_model is None:
            esm_model = self.esm_model
        if esm_batch_converter is None:
            esm_batch_converter = self.esm_batch_converter
        if esm_alphabet is None:
            esm_alphabet = self.esm_alphabet
        
        max_len = max(self.pep_lens)
        hla_seq = self.get_hla_seq(hla)
        padded_seq = self.add_padding(peptide_seq, max_len, "-")
        concat_seq = padded_seq + hla_seq
        
        #TODO: for all of these, check that it should be unsqueeze(1)
        #converted this code to be for just one peptide since that is what the dataloader needs
        if encode_choice == "embed":
            concat_processed = torch.tensor([amino_acid_mapping[aa] for aa in concat_seq])
            padding_mask = (concat_processed == amino_acid_mapping["-"]).unsqueeze(1)
            
        #TODO: check if this is correctly implemented for a one peptide at a time basis 
        elif encode_choice == "esm":
            padded_seq = self.add_padding(peptide_seq, max_len, "<pad>")
            concat_seq = padded_seq + hla_seq
            concat_input = [("Concat_Input", concat_seq)]
            concat_batch_labels, concat_batch_strs, concat_batch_tokens = esm_batch_converter(concat_input)
            #concat_batch_lens = (concat_batch_tokens != esm_alphabet.padding_idx).sum(1) #only need this if you want to later use the means instead of the token representations for each amino acid
            
            # Extract per-residue representations (on CPU)
            with torch.no_grad():
                concat_results = esm_model(concat_batch_tokens, repr_layers=[33], return_contacts=True)
            concat_processed = concat_results["representations"][33]
            #this is the padding mask for the transformer (padding for the esm model has already been incorporated into the <pad> token for esm_model to deal with - double check! TODO)
            padding_mask = (concat_batch_tokens == esm_alphabet.padding_idx).unsqueeze(1) #QUESTION/TODO: Do I need to unsqueeze this? this is currently [batch_size, seq_length]
    
        elif encode_choice == "one_hot":
            concat_processed = torch.tensor([one_hot_dict[aa] for aa in concat_seq])
            padding_bool = [aa == "-" for aa in concat_seq]
            padding_mask = torch.tensor(padding_bool).unsqueeze(1)
    
        elif encode_choice == "pc":
            concat_processed = torch.tensor([pc_dict[aa] for aa in concat_seq])
            padding_bool = [aa == "-" for aa in concat_seq]
            padding_mask = torch.tensor(padding_bool).unsqueeze(1)
        else:
            raise ValueError(f"Unsupported encoding choice: {encode_choice}. Must be 'one_hot', 'esm', embed', or 'pc'")
        
        return concat_processed, padding_mask
