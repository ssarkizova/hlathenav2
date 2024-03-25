from typing import Tuple
import warnings

from Bio import SeqIO
import pandas as pd

from hlathena.peptide_dataset import PeptideDataset


def tile_peptides(fasta_file: str, lengths: Tuple[int,...] = (8, 9, 10, 11)) -> PeptideDataset:
    lengths_set = set(lengths)
    if not (lengths_set <= {8, 9, 10, 11}):
        warnings.warn("Peptides of length outside the range 8-11 may not be supported by \
                      other HLAthena functions.")

    data = [{'pep': str(record.seq[start:start+length]), 'record_id': record.id, 'start_pos': start, 'length': length}
            for length in lengths_set
            for record in SeqIO.parse(fasta_file, "fasta")
            for start in range(0, len(record) - length + 1)]

    return PeptideDataset(pd.DataFrame(data))
