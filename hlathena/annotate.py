"""Peptide reference annotation module"""

import pandas as pd
from Bio import SeqIO
import ahocorasick
import importlib_resources
from hlathena.peptide_dataset import PeptideDataset

exp_references = importlib_resources.files('hlathena').joinpath('references').joinpath('expression')


def list_tcga_expression_references() -> pd.DataFrame:
    """List available TCGA expression reference datasets

    Returns:
        Dataframe with TCGA cancer type abbreviations and descriptions. 
    """
    ref_expr_info = str(exp_references.joinpath('tcga_abbreviations.txt'))
    return pd.read_table(ref_expr_info, sep='\t', index_col='Study Abbreviation')


def get_reference_gene_ids(  # pylint: disable=too-many-locals
    pep_df: PeptideDataset,
    ref_fasta: str = None,
    add_context: bool = True,
) -> PeptideDataset:
    """Method to annotate peptides with gene IDs.

    Given a reference FASTA file and a list of peptide sequences, this function
    identifies the gene(s) in the reference file that produce each peptide
    sequence, and returns a DataFrame with information about the peptide,
    the corresponding gene(s), and (optionally) flanking amino acid sequences.

    The default reference FASTA is the hg19 proteome.

    Args:
        pep_df: Dataframe with peptide sequences
        ref_fasta: Path to a reference fasta containing protein sequences. If None,
            a default fasta file is used.
        add_context: If True, add columns to the output DataFrame containing the
            30 amino acids upstream and downstream of each peptide sequence.

    Returns:
        A DataFrame with columns for the peptide sequence, corresponding gene(s),
        and optional flanking amino acid sequences.
    """
    seqs = pep_df.get_peptides()

    automaton = ahocorasick.Automaton()
    for idx, key in enumerate(seqs):
        automaton.add_word(key, (idx, key))

    automaton.make_automaton()

    if ref_fasta is None:
        references = importlib_resources.files('hlathena').joinpath('references')
        ref_fasta = str(references.joinpath('HUGO_proteome.fa'))
    record_dict = SeqIO.index(ref_fasta, 'fasta')

    new_df = []
    for idx, key in enumerate(record_dict):
        hugo = record_dict[key].name.split("|")[1]
        seq = str(record_dict[key].seq)
        for end_index, (_, original_value) in automaton.iter(seq):
            start_index = end_index - len(original_value) + 1
            peptide_info = [original_value, hugo]

            if add_context:
                ctx_up_start = start_index - 30 if start_index > 30 else 0
                ctx_down_end = end_index + 31 if len(seq) > end_index + 31 else len(seq)

                up_seq = seq[ctx_up_start:start_index]
                dn_seq = seq[end_index + 1:ctx_down_end]

                if len(up_seq) < 30:
                    up_seq = "-" * (30 - len(up_seq)) + up_seq
                if len(dn_seq) < 30:
                    dn_seq = dn_seq + "-" * (30 - len(dn_seq))

                peptide_info += [up_seq, dn_seq]
                if peptide_info not in new_df:
                    new_df.append(peptide_info)
            else:
                if peptide_info not in new_df:
                    new_df.append(peptide_info)
    if add_context:
        return PeptideDataset(
            pd.DataFrame(new_df, columns=['pep', 'Hugo_Symbol', 'ctex_up', 'ctex_dn'])
        )

    return PeptideDataset(pd.DataFrame(new_df, columns=['pep', 'Hugo_Symbol']))


def add_tcga_expression(
    peptide_dataset: PeptideDataset,
    cancer_type: str,
    hugo_col: str = 'Hugo_Symbol',
) -> PeptideDataset:
    """Add column with TCGA reference expression data.

    Args:
        peptide_dataset: A dataframe with peptide sequences and associated Hugo symbols.
        cancer_type: TCGA cancer type abbreviation.
        hugo_col: Name of dataframe column with Hugo symbols.

    Returns:
        DataFrame with appended TCGA expression column for selected TCGA reference
    """
    try:
        ref_expr = str(
            exp_references.joinpath(f'{cancer_type}_Differential_Gene_Expression_Table.txt')
        )
        ref_expr_df = pd.read_table(
            ref_expr,
            sep='\t',
            usecols=['Gene symbol', 'Cancer Sample Med']
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            'Expression file not found for {cancer_type}. Available cancer'
            ' types can be found using list_expression_references()'
        ) from exc

    ref_expr_df = ref_expr_df.rename(
        columns={'Cancer Sample Med': f'{cancer_type}_TPM'}  # rename new column
    )
    # merge with original dataset
    annotated_df = peptide_dataset.pep_df.merge(
        ref_expr_df,
        left_on=hugo_col,
        right_on='Gene symbol'
    ).drop(columns=['Gene symbol'])

    return PeptideDataset(annotated_df)
