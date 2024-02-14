"""Peptide dataset"""

import logging
from typing import List, Union, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from torch import Tensor
from torch.utils.data import Dataset

from hlathena.definitions import AMINO_ACIDS, INVERSE_AA_MAP, PEP_LENS
from hlathena.pep_hla_encoder import PepHLAEncoder

class PeptideDataset(Dataset):
    """Creates a peptide dataset

    Attributes:
        pep_df: Dataframe with peptides and supporting information
    """

    def __init__(  # pylint: disable=too-many-arguments, too-many-branches
        self,
        pep_df: Union[pd.DataFrame, List[str]],
        allele_name: Optional[str] = None,
        pep_col_name: Optional[str] = 'pep',
        allele_col_name: Optional[str] = None,
        target_col_name: Optional[str] = None,
        encode: Optional[bool] = False,
        reset_index: Optional[bool] = False,
    ) -> None:
        """Initializes a peptide dataset.

         Only requirement is peptide input. Optionally, supports corresponding
         allele and target columns.

        Args:
            pep_df: Dataframe with peptide column or list of peptides.
            allele_name: Name of allele.
            pep_col_name: Name of peptide column.
            allele_col_name: Name of allele column.
            target_col_name: Name of target column.
            encode: Whether to create encoded peptide list.
            reset_index: Whether to reset index.
        """
        super().__init__()

        # Create from peptide list or dataframe
        if isinstance(pep_df, list) or (np.ndim(pep_df) <= 1) or (pep_df.shape[1] == 1):
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
        if allele_col_name is not None:
            if allele_col_name not in pep_df.columns:
                raise KeyError(f"Column {allele_col_name} does not exist!")

        # If specified, check that the allele column name exist
        if target_col_name is not None:
            if target_col_name not in pep_df.columns:
                raise KeyError(f"Column {target_col_name} does not exist!")

        # Add standard columns for peptide, peptide length, and optionally allele
        if 'ha__pep' not in pep_df.columns:
            self.pep_df.insert(
                loc=0,
                column='ha__pep',
                value=self.pep_df[pep_col_name].apply(str.upper)
            )

        if 'ha__pep_len' not in pep_df.columns:
            self.pep_df.insert(
                loc=1,
                column='ha__pep_len',
                value=self.pep_df[pep_col_name].apply(len)
            )

        if (allele_name is not None) and (allele_col_name is not None):
            logging.warning(
                "Both allele_name or allele_column_name are specified; allele_name will be ignored"
            )

        if allele_col_name is not None:
            if 'ha__allele' not in pep_df.columns:
                self.pep_df.insert(
                    loc=0,
                    column='ha__allele',
                    value=self.pep_df[allele_col_name]
                )
        elif allele_name is not None:
            if 'ha__allele' not in pep_df.columns:
                self.pep_df.insert(
                    loc=0,
                    column='ha__allele',
                    value=allele_name
                )

        if target_col_name is not None:
            if 'ha__target' not in pep_df.columns:
                self.pep_df.insert(
                    loc=2,
                    column='ha__target',
                    value=self.pep_df[target_col_name]
                )

        # Check that all peptides are valid sequences.
        # TODO:  provide option to filter out invalid peps
        self._check_peps_present()
        self._check_valid_sequences()

        # TODO: Clean allele names e.g.: HLA*A01:01  -> A0101
        # TODO: supported alleles provided as sequence

        self.encode = encode
        if self.encode:
            self.encoded_peptides: List[Tensor] = (
                PepHLAEncoder.encode_peptides(self.pep_df['ha__pep'].values)
            )
            # TODO: hardcoded for one-hot only, need to get the proper dims from aafeatmap
            self.feature_dimensions = len(AMINO_ACIDS) * self.pep_df['ha__pep_len'][0]

    def __len__(self) -> int:
        """Returns the number of peptides in the dataset."""
        return self.pep_df.shape[0]

    def __getitem__(self, idx) -> Union[pd.Series, Tensor, Tuple[Tensor, int]]:
        """Fetches peptide at idx row.

        Args:
            idx: The index to retrieve the peptide from.

        Returns:
            If encode is True, returns the encoded peptide as a Tensor. Otherwise, returns
            a Series with the peptide and its associated features.
        """
        if not self.encode:
            return self.pep_df.iloc[idx]  # TODO: what do we want this to do?
        if 'ha__target' not in self.pep_df.columns:
            return self.encoded_peptides[idx]

        return self.encoded_peptides[idx], self.pep_df['ha__target'][idx]

    def get_peptides(self) -> List[str]:
        """Get all peptides in the dataset

        Returns:
            List of peptides
        """
        return self.pep_df['ha__pep'].values.tolist()

    def get_peptide_lengths(self) -> List[int]:
        """Get unique peptide lengths in dataset

        Returns:
            List of unique peptide lengths
        """
        return np.unique(self.pep_df['ha__pep_len']).tolist()

    def get_alleles(self) -> Optional[List[str]]:
        """Get unique alleles in the dataset

        Returns:
            List of unique alleles
        """
        if 'ha__allele' in self.pep_df.columns:
            return np.unique(self.pep_df['ha__allele']).tolist()

        return None

    def get_allele_peptide_counts(self) -> Optional[pd.DataFrame]:
        """Get count of peptides per allele

        Returns:
            Peptide counts per allele
        """
        if 'ha__allele' in self.pep_df.columns:
            tbl = pd.DataFrame(self.pep_df.groupby(["ha__allele"]).size())
            tbl.columns = ['n_pep']
            return tbl

        return None

    def get_allele_peptide_length_counts(self) -> Optional[pd.DataFrame]:
        """Get count of peptides per length

        Returns:
            Counts of unique peptide lengths
        """
        if 'ha__allele' in self.pep_df.columns:
            tbl = self.pep_df.groupby(["ha__allele", 'ha__pep_len']).size()
            tbl = tbl.unstack(level='ha__pep_len').sort_index().fillna(0).astype(int)
            return tbl

        return None

    def _check_peps_present(self) -> None:
        """Check that there are peptides in the peptide set

        Raises:
            Exception: No peptide sequences provided
        """
        if not len(self) > 0:
            raise ValueError("No peptide sequences provided")

    def _check_valid_sequences(self) -> None:
        """Check that peptides are composed of the standard 20 amino acids"""
        try:
            for pep in self._split_peptides():
                for aa in pep:  # pylint: disable=invalid-name
                    assert aa in AMINO_ACIDS
        except AssertionError:
            warnings.warn(
                "Peptides sequences contain invalid characters. \
                 Please ensure peptides sequences only contain the \
                 20 standard amino acid symbols."
            )

    def _check_supported_peptide_lengths(self) -> None:
        """Check that peptide lengths are valid lengths

        Raises:
            Exception: Peptides lengths are not in the allowed range (8-12).
        """
        if not all(pep_len in PEP_LENS for pep_len in self.get_peptide_lengths()):
            raise ValueError("Peptides lengths are not in the allowed range (8-12).")

    def _check_same_peptide_lengths(self) -> None:
        """Check that all peptides are the same length

        Raises:
            Exception: Peptides of multiple lengths are present in the dataset
        """
        if len(self.get_peptide_lengths()) != 1:
            raise ValueError(
                f"Peptides of multiple lengths are present in the dataset: \
                {self.get_peptide_lengths()}"
            )

    def _split_peptides(self) -> List[List[str]]:
        """Split peptides into list of list of amino acids"""
        return [list(pep) for pep in self.get_peptides()]

    def subset_data(
        self,
        peplens: Optional[Union[int, List[int]]] = None,
        alleles: Optional[Union[str, List[str]]] = None,
        reset_index: Optional[bool] = False,
    ) -> None:
        """Subset the peptide dataset to specified alleles and lengths.

        Args:
            peplens: Length of peptides to include in dataset
            alleles: Name of allele to include in dataset based on entries in
                allele` column of pep_df (`allele` column only required if this
                parameter is specified)
            reset_index: Whether to reset the index of the dataframe
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

    def decode_peptide(self, encoded: Tensor) -> Union[List[str], str]:
        """Decode peptide tensor and return peptide
        
        Args:
            encoded: Tensor with encoded peptide
        
        Returns:
            Decoded peptide string
        """
        # TODO: add this back to accommodate for peptide-level features
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
