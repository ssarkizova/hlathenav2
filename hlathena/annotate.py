""" Peptide reference annotation module  """

import pandas as pd
from Bio import SeqIO
import ahocorasick
import importlib_resources

from hlathena import references

exp_references = importlib_resources.files('hlathena').joinpath('references').joinpath('expression')

def list_expression_references() -> pd.DataFrame:
    """
    Returns:
        Dataframe with TCGA cancer type abbreviations and descriptions. 
    """
    ref_expr_info = str(exp_references.joinpath(f'tcga_abbreviations.txt'))
    return pd.read_table(ref_expr_info, sep='\t', index_col='Study Abbreviation')
    
    
def add_tcga_expression(pep_df: pd.DataFrame, cancer_type: str, hugo_col: str = 'Hugo_Symbol') -> pd.DataFrame:
    """Add column with TCGA reference expression data.

    Args:
        pep_df (pd.DataFrame): A dataframe with peptide sequences and associated Hugo symbols.
        cancer_type (str): TCGA cancer type abbreviation.
        hugo_col (str): Name of dataframe column with Hugo symbols.

    Returns:
        pd.DataFrame with TCGA expression column
    """
    try:
        ref_expr = str(exp_references.joinpath(f'{cancer_type}_Differential_Gene_Expression_Table.txt'))
    except:
        print("Expression file not found for {cancer_type}. \
               Available cancer types can be found using list_expression_references()")
    
    ref_expr_df = pd.read_table(ref_expr, sep='\t', usecols=['Gene symbol', 'Cancer Sample Med'])
    
    annotated_df = pep_df.merge(ref_expr_df, left_on='Hugo_Symbol', right_on='Gene symbol').drop(columns=['Gene symbol'])
    annotated_df = annotated_df.rename(columns={"Cancer Sample Med": f"{cancer_type}_TPM"})
    return annotated_df