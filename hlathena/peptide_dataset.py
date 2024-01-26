""" Featurized peptide dataset """

import logging
from typing import List, Tuple, Union, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from torch import Tensor
from torch.utils.data import Dataset

# from hlathena import references # cleo note: causing circular import error, need to remove or rearrange imports
from hlathena.definitions import AMINO_ACIDS, AMINO_ACIDS_EXT, INVERSE_AA_MAP, PEP_LENS
from hlathena.pep_encoder import PepEncoder


## TO DO - check if the the output signatures are well defined... e.g. can None pass for a List[str]?

class PeptideDataset(Dataset):
    """
    Creates an a peptide dataset of hit and decoy binders which are featurized

    Attributes:
        peptide_feats_df (pd.DataFrame): dataframe with peptide features for training
        peptide_features_dim (int): count of peptide features
        peptides (np.ndarray): list of hit and decoy peptide
        #binds (np.ndarray): list of target value for peptides
        peptide_length (int): length of peptides in dataset

        aa_feature_map (AminoAcidFeatureMap): amino acid feature dataframe for training
        feature_dimensions (int): count of total features per peptide
        encoded_peptides (List[Tensor]): list of feature tensors for training

    """

    def __init__(
        self,
        pep_df: pd.DataFrame,
        allele_name: Optional[str] = None,
        pep_col_name: Optional[str] = 'pep',
        allele_col_name: Optional[str] = None,        
        target_col_name: Optional[str] = None, # TO DO: target_col or target_value, also support this to be single number 1 or 0?
        encode: Optional[bool] = False,
        reset_index: Optional[bool] = False) -> None:
        """ Inits a peptide dataset which featurizes the hit and decoy peptides.

        Args:
            pep_df (pd.DataFrame): list of peptides
           
            label_col (str, optional): name of column containing target labels (default is None)
            aa_featurefiles (List[os.PathLike], optional): list of feature matrix files (default one-hot encoding)
            feat_cols (List[str], optional): list of columns in the input dataframe (pep_df) 
                                             which are to be interpreted as peptide features for training/prediction purposees (e.g. 'TPM')
        """
        super().__init__()
        
        # Create from peptide list or dataframe
        if(np.ndim(pep_df)<=1) or (pep_df.shape[1]==1): 
            # Assume list of peptides provided
            pep_df = pd.DataFrame(pep_df)
            pep_df.columns = ['pep']

        self.pep_df = pep_df.copy()
        if reset_index:
            self.pep_df.reset_index(drop=True, inplace=True)

        # Check that the specified peptide column name exist
        if pep_col_name not in pep_df.columns:
            raise KeyError(f"Column {pep_col_name} does not exist!")
        
        # If specified, check that the allele column name exist            
        if not allele_col_name is None:
            if allele_col_name not in pep_df.columns:
                raise KeyError(f"Column {allele_col_name} does not exist!")

        # If specified, check that the allele column name exist
        if not target_col_name is None:
            if target_col_name not in pep_df.columns:
                raise KeyError(f"Column {target_col_name} does not exist!")


        ### TO DO - add a check to make sure that the 'reserved' column names we will be adding 
        ### are not present in the dataframe with some user-specific values

        # Add standard columns for peptide, peptide length, and optionally allele
        if not 'ha__pep' in pep_df.columns:
            self.pep_df.insert(
                loc=0,
                column='ha__pep',
                value=self.pep_df[pep_col_name].apply(str.upper))
            
        if not 'ha__pep_len' in pep_df.columns:
            self.pep_df.insert(
                loc=1,
                column='ha__pep_len',
                value=self.pep_df[pep_col_name].apply(len))

        if (not allele_name is None) and (not allele_col_name is None):
            logging.warning(f"Both allele_name or allele_column_name are specified; allele_name will be ignored")

        if not allele_col_name is None:
            if not 'ha__allele' in pep_df.columns:
                self.pep_df.insert(
                    loc=0,
                    column='ha__allele',
                    value=self.pep_df[allele_col_name])
        elif not allele_name is None:
            if not 'ha__allele' in pep_df.columns:
                self.pep_df.insert(
                    loc=0,
                    column='ha__allele',
                    value=allele_name)                
        
        if not target_col_name is None:
            if not 'ha__target' in pep_df.columns:
                self.pep_df.insert(
                        loc=2,
                        column='ha__target',
                        value=self.pep_df[target_col_name])


        # Check that all pepetides are valid sequences. TO DO - provide option to filter out invalid peps
        self._check_peps_present()
        self._check_valid_sequences()

        # TO DO: We want to be able to keep the longer peptides so that we can plot the length distribution 
        # and maybe for other analyses as well. Skip this check now, option to filter provided later
        # self._check_same_peptide_lengths() 

        if 'ha__allele' in pep_df.columns:
            self.format_allele_names() # TO DO: E.g.: HLA*A01:01  -> A0101; might be good to save an output file with the name changes that we made -> Logger
            self._check_alleles()      # TO DO: supported pre-processed alleles vs alleles provided as sequence... 


        # SISI - sub below with the pep_encoder and hla_encoder class. something like...:
        if encode:
            self.encoded_peptides: List[Tensor] = PepEncoder._encode_peptides(self.pep_df['ha__pep'].values)
            # self.encoded_alleles: List[Tensor] = HLAEncoder.encode(pep_df['ha__allele'])
            self.feature_dimensions = len(AMINO_ACIDS_EXT)*self.pep_df['ha__pep_len'][0] # TO DO harcoded for one-hot only, need to get the proper dims from aafeatmap
        # Init peptide features dataframe
        # self.peptide_feats_df = self.pep_df[['ha__pep'] + feat_cols].copy()
        # self.peptide_feats_df.drop(columns='ha__pep', inplace=True)    # hmm why?, is it because we are making features?
        # self.peptide_features_dim = len(list(self.peptide_feats_df.columns))
        #
        # if self.peptide_features_dim:
        #    print()
        #    print(f'Peptide set features: {list(self.peptide_feats_df.columns)}')
        #    print(self.peptide_feats_df.describe())
        #    print()
        #
        #self.aa_feature_map: AminoAcidFeatureMap = AminoAcidFeatureMap(featurefiles=aa_featurefiles)
        # self.feature_dimensions: int = (self.aa_feature_map.feature_count * self.peptide_length) \
        #                               + self.peptide_features_dim

        # From: get_aa_encoded_peptide_map() - TO DO - ADD BACK THE peptide-level features
        #if self.peptide_features_dim:
        #    peps_aafeatmat = pd.concat(
        #        [peps_aafeatmat, self.peptide_feats_df], axis=1)
        ## TO DO: separate out aa encoding and peptide encoding
        ## TO DO: keep peps and pep fts in same df

    def __len__(self) -> int:
        return self.pep_df.shape[0]

    # TO DO - does the output spec work if only one thing is retured but the -> says tuple?
    def __getitem__(self, idx) -> Tuple[Tensor, float]:
        if 'ha__target' not in self.pep_df.columns:
            return self.encoded_peptides[idx]
        else:
            return self.encoded_peptides[idx], self.pep_df['ha__target'][idx]
        
    def get_peptides(self) -> List[str]:
        return self.pep_df['ha__pep'].values

    def get_peptide_lengths(self) -> List[int]:
        return np.unique(self.pep_df['ha__pep_len'])
    
    def get_alleles(self) -> List[str]:
        if 'ha__allele' in self.pep_df.columns:
            return np.unique(self.pep_df['ha__allele'])
        else: 
            return None
    
    def get_allele_peptide_counts(self) -> Union[pd.DataFrame, None]:
        if 'ha__allele' in self.pep_df.columns:
            tbl = pd.DataFrame(self.pep_df.groupby(["allele"]).size())
            tbl.columns = ['n_pep']
            return tbl
        else: 
            return None
    
    def get_allele_peptide_length_counts(self) -> Union[List[str], None]:
        if 'ha__allele' in self.pep_df.columns:
            tbl = self.pep_df.groupby(["ha__allele", 'ha__pep_len']).size()
            tbl = tbl.unstack(level='ha__pep_len').sort_index().fillna(0).astype(int)
            return tbl
        else: 
            return None

    def _check_peps_present(self) -> None:
        """
        Check that there are peptides in the peptide set

        Raises:
            Exception: No peptide sequences provided
        """
        if not len(self) > 0:
            raise Exception("No peptide sequences provided")

    def _check_valid_sequences(self) -> None:
        """
        Check that peptides are composed of the standard 20 amino acids
        """
        try:
            for pep in self._split_peptides():
                for aa in pep:
                    assert aa in AMINO_ACIDS
        except AssertionError:
            print("Peptides sequences contain invalid characters. \
                   Please ensure peptides sequences only contain the 20 standard amino acid symbols.") 

    def _check_supported_peptide_lengths(self) -> None:
        """
        Check that peptide lengths are valid lengths
        """
        try:
            assert all(pep_len in PEP_LENS for pep_len in self.get_peptide_lengths())

        except AssertionError:
            print("Peptides lengths are not in the allowed range (8-12).")

    def _check_same_peptide_lengths(self) -> None:
        """
        Check that all peptides are the same length
        """
        assert len(self.get_peptide_lengths()) == 1, f"Peptides of multiple lengths are present in the dataset: {self.get_peptide_lengths()}"

    # TO DO: we should also check if the alleles are valid
    # (format names and and/or look at the full sequence is provided)
    def _check_alleles(self) -> None:
        pass
        
    # TO DO: similarly to the Terra workflow, add some code to standardize allele names; Perhaps this belongs in a generic 'tools' .py
    def format_allele_names(self) -> None:
        pass
        
    def _split_peptides(self) -> List[List[str]]:
        """
        Split peptides into list of list of amino acids
        """
        return [list(pep) for pep in self.get_peptides()]

    def subset_data(
            self,
            peplens: Optional[Union[int, List[int], str, List[str]]] = None,
            alleles: Optional[Union[str, List[str]]] = None,
            reset_index: Optional[bool] = False
            ) -> None:
        """ Subset the peptide dataset to specified alleles and lengths.

        Args:
            peplens: int or List[int], optional, default: None
                length of peptides to include in dataset
            alleles:  str or List[str], optional, default: None
                name of allele to include in dataset 
                based on entries in `allele` column of pep_df 
                (`allele` column only required if this parameter is specified)
            reset_index: bool, optional, default: False
        """
        # Filter to specified peptide length(s) if any
        if peplens is not None:
            if not is_list_like(peplens):
                peplens = [peplens]
            self.pep_df = self.pep_df[self.pep_df['ha__pep_len'].isin(peplens)]

        # Filter to specified allele(s) if any
        if alleles is not None:
            if 'ha__allele' not in self.pep_df.columns:
                # TO DO: warning message that there is no allele column in the dataset to subset on
                pass
            else:
                if not is_list_like(alleles):
                    alleles = [alleles]
                self.pep_df = self.pep_df[self.pep_df['ha__allele'].isin(alleles)]

        if reset_index:
            self.pep_df.reset_index()
        
    # TO DO - keep here or in PepEncoder?? Also edit hard coded values!
    def decode_peptide(self, encoded: Tensor) -> List[str]:
        """Decode peptide tensor and return peptide
        
        Args:
            encoded (Tensor): encoded peptide
        
        Returns:
            str: decoded peptide
        """
        
        # TO DO - add this back to accomodate for peptide-level features
        # if self.peptide_features_dim > 0:
        #    encoded = encoded[:-self.peptide_features_dim]
        
        encoded = encoded.reshape(
            # self.peptide_length, self.aa_feature_map.feature_count)
            9, 20)
        dense = encoded.argmax(-1)
        if len(dense.shape) > 1:
            peptide = [''.join([INVERSE_AA_MAP[aa.item()]
                               for aa in p]) for p in dense]
        else:
            peptide = ''.join([INVERSE_AA_MAP[aa.item()] for aa in dense])
        return peptide