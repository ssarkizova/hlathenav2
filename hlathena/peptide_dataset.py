""" Featurized peptide dataset """

import os
from typing import List, Tuple
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing  # Peptide encoding
import torch
from torch import Tensor
from torch.utils.data import Dataset
from Bio import SeqIO
import ahocorasick
import importlib_resources

from hlathena import references
from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, INVERSE_AA_MAP


class PeptideDataset(Dataset):
    """
    Creates an a peptide dataset of hit and decoy binders which are featurized

    Attributes:
        peptide_feats_df (pd.DataFrame): dataframe with peptide features for training
        peptide_features_dim (int): count of peptide features
        peptides (np.ndarray): list of hit and decoy peptide
        binds (np.ndarray): list of target value for peptides
        peptide_length (int): length of peptides in dataset
        aa_feature_map (AminoAcidFeatureMap): amino acid feature dataframe for training
        feature_dimensions (int): count of total features per peptide
        encoded_peptides (List[Tensor]): list of feature tensors for training

    """

    def __init__(self, \
                 pep_df: pd.DataFrame, \
                 peplen: int=None, \
                 allele: str=None, \
                 label_col: str=None, \
                 aa_featurefiles: List[os.PathLike]=None, \
                 feat_cols: List[str]=None) -> None:
        """ Inits a peptide dataset which featurizes the hit and decoy peptides.

        Args:
            pep_df (pd.DataFrame): list of peptides
            peplen (int, optional): length of peptides to include in dataset
            allele (str, optional): name of allele in `allele` column of pep_df 
                                        (`allele` column only required if this parameter is specified)
            label_col (str, optional): name of column containing target labels (default is None)
            aa_featurefiles (List[os.PathLike], optional): list of feature matrix files (default one-hot encoding)
            feat_cols (List[str], optional): list of peptide feature columns
        """
        super().__init__()
        
        if peplen != None:
            pep_df['length'] = [len(pep) for pep in pep_df['seq']]
            pep_df = pep_df[pep_df['length']==peplen].copy()
        
        if allele != None:
            pep_df = pep_df[pep_df['allele']==allele].copy()
        
        self.peptide_feats_df = pep_df.copy()
        self.peptide_feats_df.index = self.peptide_feats_df['seq']
        self.peptide_feats_df.drop(columns="seq", inplace=True)

        feat_cols = [] if feat_cols is None else feat_cols
        self.peptide_feats_df = self.peptide_feats_df[feat_cols]

        self.peptide_features_dim = len(list(self.peptide_feats_df.columns))

        if self.peptide_features_dim:
            print()
            print(f'Peptide set features: {list(self.peptide_feats_df.columns)}')
            print(self.peptide_feats_df.describe())
            print()
            
        self.peptides = np.asarray(pep_df['seq'].values)
        self.peptide_length = self._set_peptide_length()

        self.aa_feature_map: AminoAcidFeatureMap = AminoAcidFeatureMap(featurefiles=aa_featurefiles)
        self.feature_dimensions: int = (self.aa_feature_map.feature_count * self.peptide_length) \
                                       + self.peptide_features_dim

        self.encoded_peptides: List[Tensor] = self._encode_peptides()

        self.binds = None
        if not label_col is None:
            self.binds = torch.from_numpy((pep_df[label_col].to_numpy()))


    def __len__(self) -> int:
        return len(self.peptides)


    def __getitem__(self, idx) -> Tuple[Tensor, float]:
        if self.binds is None:
            return self.encoded_peptides[idx]
        else:
            return self.encoded_peptides[idx], self.binds[idx]


    def _set_peptide_length(self) -> None:
        """
        Set peptide length if peptides are valid sequences & are all the same length
        """
        pep_len = None
        try:
            self._check_valid_lengths()
            self._check_peps_present()
            self._check_valid_sequences()
            pep_len = len(self.peptides[0])
            assert(all(len(pep)==pep_len for pep in self.peptides))
            
        except AssertionError:
            print("Peptides are different lengths. Peptide lengths must be equal.")
        
        return pep_len
        

    
    def _check_valid_lengths(self) -> None:
        """Check that all peptides are the same length
        """
        if len(self.peptides) > 0:
            pep_len = len(self.peptides[0])
            assert(all(len(pep)==pep_len for pep in self.peptides))

    def _check_peps_present(self) -> None:
        """Check that there are peptides in the peptide set

        Raises:
            Exception: No peptide sequences provided
        """
        if not len(self.peptides) > 0:
            raise Exception("No peptide sequences provided")

    def _check_valid_sequences(self) -> None:
        """Check that the peptide is composed of valid amino acids
        """
        for pep in self._split_peptides():
            for aa in pep:
                assert aa in AMINO_ACIDS
                
    def _split_peptides(self) -> List[List[str]]:
        """Split peptides into list of list of amino acids
        """
        return [list(seq) for seq in self.peptides]
    

    def _encode_peptides(self) -> List[Tensor]:
        """Return featurized peptides
        
        Returns:
            List[Tensor]: featurized peptide tensors
        """
        encoding_map: pd.DataFrame = self.get_aa_encoded_peptide_map()
        encoded = []

        # TODO: is there a simpler way of creating tensors.. pytorch lightning?
        for i in range(len(self.peptides)):
            encoded_peptide: Tensor = torch.as_tensor(encoding_map.iloc[i].values).float()
            encoded.append(encoded_peptide)
        return encoded


    def encode_onehot(self) -> np.ndarray:
        """One hot encode peptides

        Returns:
            np.ndarray: one hot encoded peptide set
        """
        ### Peptide encoding
        peps_split = self._split_peptides()
        encoder = sklearn.preprocessing.OneHotEncoder(
            categories=[AMINO_ACIDS] * self.peptide_length)
        encoder.fit(peps_split)
        encoded = encoder.transform(peps_split).toarray()
        return encoded


    def get_aa_encoded_peptide_map(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: Amino acid featurization of peptides in the class
        """
        onehot_encoded: np.ndarray = self.encode_onehot()

        onehot_only = not self.aa_feature_map.feature_files

        if onehot_only:
            peps_aafeatmat: pd.DataFrame = pd.DataFrame(onehot_encoded, index=self.peptides)
        else:
            aamap = self.aa_feature_map.feature_map
            # Block diagonal aafeatmat
            aafeatmat_bd: np.ndarray = np.kron(np.eye(self.peptide_length, dtype=int), aamap)
            # Feature encoding (@ matrix multiplication)
            feat_names: List[List[str]] = list(np.concatenate( \
                                          [(f'p{i+1}_' + aamap.columns.values).tolist() \
                                                for i in range(self.peptide_length)]).flat)
            peps_aafeatmat: pd.DataFrame = pd.DataFrame(onehot_encoded @ aafeatmat_bd, \
                                                        columns=feat_names, \
                                                        index=self.peptides)

        if self.peptide_features_dim:
            peps_aafeatmat = pd.concat([peps_aafeatmat, self.peptide_feats_df], axis=1)
# TODO: separate out aa encoding and peptide encoding
# TODO: keep peps and pep fts in same df
        return peps_aafeatmat


    def decode_peptide(self, encoded: Tensor) -> List[str]:
        """Decode peptide tensor and return peptide

        Args:
            encoded (Tensor): encoded peptide

        Returns:
            str: decoded peptide
        """
        if self.peptide_features_dim > 0:
            encoded = encoded[:-self.peptide_features_dim]

        encoded = encoded.reshape(self.peptide_length, self.aa_feature_map.feature_count)
        dense = encoded.argmax(-1)
        if len(dense.shape) > 1:
            peptide = [''.join([INVERSE_AA_MAP[aa.item()] for aa in p]) for p in dense]
        else:
            peptide = ''.join([INVERSE_AA_MAP[aa.item()] for aa in dense])
        return peptide
    
    
    def get_reference_gene_ids(self, \
                               ref_fasta: str = None, \
                               add_context: bool = True):
        """
        Given a reference FASTA file and a list of peptide sequences, this function identifies the gene(s) 
        in the reference file that produce each peptide sequence, and returns a DataFrame with information 
        about the peptide, the corresponding gene(s), and (optionally) flanking amino acid sequences.

        Args:
            ref_fasta (str): Path to a reference fasta containing protein sequences. If None, 
                a default fasta file is used.
            add_context (bool): If True, add columns to the output DataFrame containing the 30 amino acids 
                upstream and downstream of each peptide sequence.

        Returns:
            A DataFrame with columns for the peptide sequence, corresponding gene(s), 
                and optional flanking amino acid sequences.
        """
        seqs = self.peptides

        automaton = ahocorasick.Automaton()
        for idx, key in enumerate(seqs):
            automaton.add_word(key, (idx, key))

        automaton.make_automaton()
        
        if ref_fasta is None:
            references = importlib_resources.files('hlathena').joinpath('references')
            ref_fasta = str(references.joinpath(f'HUGO_proteome.fa'))
        record_dict = SeqIO.index(ref_fasta, "fasta")
        
        new_df = []
        for idx, key in enumerate(record_dict):
            hugo = record_dict[key].name.split("|")[1]
            seq = str(record_dict[key].seq)
            for end_index, (insert_order, original_value) in automaton.iter(seq):
                start_index = end_index - len(original_value) + 1
                peptide_info = [original_value, hugo]

                if add_context:
                    ctx_up_start = start_index - 30 if start_index > 30 else 0
                    ctx_down_end = end_index + 30 if len(seq) > end_index + 30 else len(seq)

                    up_seq = seq[ctx_up_start:start_index]
                    dn_seq = seq[end_index+1:ctx_down_end]

                    if len(up_seq) < 30:
                        up_seq = "-"*(30-len(up_seq)) + up_seq
                    if len(dn_seq) < 30:
                        dn_seq = dn_seq + "-"*(30-len(dn_seq))

                    peptide_info += [up_seq, dn_seq]
                    if not peptide_info in new_df:
                        new_df.append(peptide_info)
                else:
                    if not peptide_info in new_df:
                        new_df.append(peptide_info)
        if add_context:
            return pd.DataFrame(new_df, columns=['seq','Hugo_Symbol','ctex_up','ctex_dn'])
        else:
            return pd.DataFrame(new_df, columns=['seq','Hugo_Symbol'])