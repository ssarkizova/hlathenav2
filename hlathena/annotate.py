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


def get_reference_gene_ids(pep_df: pd.DataFrame, \
                            pep_col: str = 'seq', \
                            ref_fasta: str = None, \
                            add_context: bool = True):
        """
        Given a reference FASTA file and a list of peptide sequences, this function identifies the gene(s) 
        in the reference file that produce each peptide sequence, and returns a DataFrame with information 
        about the peptide, the corresponding gene(s), and (optionally) flanking amino acid sequences.

        Args:
            pep_df (pd.DataFrame): Dataframe with peptide sequences
            pep_col (str): Name of peptide column
            ref_fasta (str): Path to a reference fasta containing protein sequences. If None, 
                a default fasta file is used.
            add_context (bool): If True, add columns to the output DataFrame containing the 30 amino acids 
                upstream and downstream of each peptide sequence.

        Returns:
            A DataFrame with columns for the peptide sequence, corresponding gene(s), 
                and optional flanking amino acid sequences.
        """
        seqs = list(pep_df[pep_col])

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