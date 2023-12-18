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
                 aa_feature_files: List[os.PathLike]=None,
                 feat_cols: List[str]=None,
                 folds: int = None) -> None:
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
                         allele_col_name=allele_col_name)
        
        if target_col_name is None:
            raise AttributeError("Missing parameter target_col_name for PepHLADataset init")
        else:
            if not 'ha__target' in pep_df.columns:
                self.pep_df.insert(
                        loc=2,
                        column='ha__target',
                        value=self.pep_df[target_col_name])
                
        self.pep_len = self._set_peptide_length()
        
        self.peptide_feature_cols = [] if feat_cols is None else feat_cols
        
        self.PepHLAEncoder = PepHLAEncoder(self.pep_len, aa_feature_files=aa_feature_files)
        
        if folds:
            self.reassign_folds(folds)
            
        
    def __getitem__(self, i) -> Tuple[Tensor, float]:
        return self.PepHLAEncoder.encode(self.pep_df.at[i, 'ha__pep'], 
                                         self.pep_df.at[i, 'ha__allele'],
                                         self.pep_df[self.peptide_feature_cols].iloc[i].values
                                        ), self.pep_df.at[i, 'ha__target'], i
                
                
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
        return self.pep_df.folds
    
    
    def reassign_folds(self, folds: int, seed: int = 1):
        np.random.seed(seed)
        self.pep_df["folds"] = np.random.choice(range(folds), len(self.pep_df), replace=True)
        
    
    def get_train_idxs(self, fold: int, decoy_mul: int = 1, resampling_hits=False, seed: int = 1) -> List[int]:
        np.random.seed(seed)
        train_idxs = [i for i,f in enumerate(self.folds()) if f!=fold]
        
        decoy_train_idxs, hit_train_idxs = [i for i in train_idxs if self.binds[i]==0], [i for i in train_idxs if self.binds[i]==1]
        decoy_train_idxs = np.random.choice(decoy_train_idxs, len(hit_train_idxs)*decoy_mul, replace=True)
        
        print(f"Resampling hits: {resampling_hits}")
        if resampling_hits:
            hit_train_idxs = hit_train_idxs * decoy_mul
        
        return list(decoy_train_idxs) + list(hit_train_idxs)
    
    
    def get_test_idxs(self, fold: int, decoy_ratio: int = 1, seed: int = 1):
        np.random.seed(seed)
        test_idxs = [i for i,f in enumerate(self.folds()) if f==fold]
        decoy_test_idxs, hit_test_idxs = [i for i in test_idxs if self.binds[i]==0], [i for i in test_idxs if self.binds[i]==1]
        decoy_test_idxs = np.random.choice(decoy_test_idxs, len(hit_test_idxs)*decoy_ratio, replace=True)
        
        return list(decoy_test_idxs) + list(hit_test_idxs)
    
    
    def _check_same_peptide_lengths(self) -> None:
        """Check that all peptides are the same length
        """
        assert(len(self.get_peptide_lengths())==1)

    
    def _set_peptide_length(self) -> None:
        """
        Set peptide length if peptides are valid sequences & are all the same length
        """
        try:
            self._check_same_peptide_lengths()
            pep_len = self.get_peptide_lengths()[0]
            return pep_len
        except AssertionError:
            print("Peptides are different lengths. Peptide lengths must be equal.")
            
    