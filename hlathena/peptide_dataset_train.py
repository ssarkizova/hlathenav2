""" Featurized peptide dataset """

import os
import re
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing  # Peptide encoding
import torch
from torch import Tensor
from torch.utils.data import Dataset

from hlathena.amino_acid_feature_map import AminoAcidFeatureMap
from hlathena.definitions import AMINO_ACIDS, INVERSE_AA_MAP
from hlathena.pep_hla_encoder import PepHLAEncoder
from hlathena.peptide_dataset import PeptideDataset


class PeptideDatasetTrain(PeptideDataset, Dataset):
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

    def __init__(self,
                 pep_df: pd.DataFrame,
                 pep_col_name: str = 'pep',
                 allele_col_name: Optional[str] = None,
                 allele_name: Optional[str] = None,
                 target_col_name: Optional[str] = None,
                 feat_cols: List[str] = None,
                 fold_col_name: Optional[str] = None,
                 folds: Optional[int] = None,
                 aa_feature_files: List[os.PathLike] = None,
                 hla_encoding_file: Optional[os.PathLike] = None,
                 reset_index: Optional[bool] = True) -> None:
        """ Inits a peptide dataset which featurizes the hit and decoy peptides.

        Args:
            hits_df (pd.DataFrame): list of binding peptides
            decoys_df (pd.DataFrame): list of decoy peptides
            aa_featurefiles (List[os.PathLike]): list of feature matrix files
            feat_cols (List[str]): list of peptide feature columns
        """

        super().__init__(pep_df=pep_df,
                         pep_col_name=pep_col_name,
                         allele_name=allele_name,
                         allele_col_name=allele_col_name,
                         target_col_name=target_col_name,
                         reset_index=reset_index)

        if target_col_name is None:
            raise AttributeError("Missing parameter target_col_name for PepHLADataset init")

        if fold_col_name is not None:
            if fold_col_name not in pep_df.columns:
                raise KeyError(f"Column {fold_col_name} does not exist!")
            if 'ha__fold' not in pep_df.columns:
                self.pep_df.insert(
                    loc=3,
                    column='ha__fold',
                    value=self.pep_df[fold_col_name])
        else:
            if 'ha__fold' not in pep_df.columns:
                self.pep_df.insert(
                    loc=3,
                    column='ha__fold',
                    value=0)
            if folds is not None:
                self.reassign_folds(folds=folds)

        self.max_len = self._set_max_length()
        # TODO: add a check to make sure the columns are present

        self.peptide_feature_cols = [] if feat_cols is None else feat_cols

        self.PepHLAEncoder = PepHLAEncoder(maxlen=self.max_len,
                                           aa_feature_files=aa_feature_files,
                                           hla_encoding_file=hla_encoding_file,
                                           is_pan_allele=len(self.get_alleles()) > 1)

    def __getitem__(self, i) -> Tuple[Tensor, float, int]:
        return (self.PepHLAEncoder.
                    encode(self.pep_at(i), self.allele_at(i), self.pep_features_at(i)),
                    self.tgt_at(i),
                    i)

    def pep_at(self, i: int) -> str:
        return self.pep_df.at[i, 'ha__pep']

    def allele_at(self, i: int) -> str:
        return self.pep_df.at[i, 'ha__allele']

    def pep_features_at(self, i: int) -> np.ndarray:
        return self.pep_df[self.peptide_feature_cols].iloc[i].values

    def tgt_at(self, i: int) -> int:
        return self.pep_df.at[i, 'ha__target']

    # def clean_allele_names(self):
    #     replace_regex = '[\*]|(HLA[-|*])|[:]|([N|Q]$)|[ ]|[-]'
    #     alleles = self.pep_df['ha__allele']
    #     alleles_clean = [re.sub(replace_regex, "", a) for a in alleles]
    #     return alleles_clean

    def set_feat_cols(self, feat_cols: List[str] = []):
        self.peptide_feature_cols = feat_cols

    def feature_dimensions(self):
        return self.PepHLAEncoder.feature_dimensions + self.peptide_features_dim()

    def peptide_features_dim(self):
        return len(self.peptide_feature_cols)

    def folds(self):
        return self.pep_df['ha__fold']

    def reassign_folds(self, folds: int, seed: int = 1):
        np.random.seed(seed)
        self.pep_df['ha__fold'] = np.random.choice(range(folds), len(self.pep_df), replace=True)

    def get_train_idxs(self, fold: int, decoy_mul: int = 1, resampling_hits=False, seed: int = 1) -> List[int]:
        np.random.seed(seed)
        train_idxs = [i for i, f in enumerate(self.folds()) if f != fold]

        decoy_train_idxs, hit_train_idxs = ([i for i in train_idxs if self.pep_df.at[i, 'ha__target'] == 0],
                                            [i for i in train_idxs if self.pep_df.at[i, 'ha__target'] == 1])
        decoy_train_idxs = np.random.choice(decoy_train_idxs, len(hit_train_idxs) * decoy_mul, replace=True)

        if resampling_hits:
            hit_train_idxs = hit_train_idxs * decoy_mul

        return list(decoy_train_idxs) + list(hit_train_idxs)

    def get_test_idxs(self, fold: int, decoy_ratio: int = 1, seed: int = 1):
        np.random.seed(seed)
        test_idxs = [i for i, f in enumerate(self.folds()) if f == fold]
        decoy_test_idxs, hit_test_idxs = ([i for i in test_idxs if self.pep_df.at[i, 'ha__target'] == 0],
                                          [i for i in test_idxs if self.pep_df.at[i, 'ha__target'] == 1])
        decoy_test_idxs = np.random.choice(decoy_test_idxs, len(hit_test_idxs) * decoy_ratio, replace=True)

        return list(decoy_test_idxs) + list(hit_test_idxs)

    def _set_max_length(self):
        return max(self.get_peptide_lengths())
