.. automodule:: hlathena
    :noindex:
    
API
===

*hlathena* can be used as a python package. Import hlathena by::

    import hlathena as ha
   
Plotting Tools
---------------

.. autosummary::
    :toctree: .

    hlathena.plotting.plot_length
    hlathena.plotting.plot_logo
    hlathena.plotting.plot_umap
    hlathena.plotting.plot_clustered_umap

Peptide Projection Tools
---------------

.. autosummary::
    :toctree: .

    hlathena.peptide_projection.PCA_encode
    hlathena.peptide_projection.get_umap_embedding
    hlathena.peptide_projection.get_peptide_clustering


Annotation Tools
---------------

.. autosummary::
    :toctree: .

    hlathena.annotate.list_tcga_expression_references
    hlathena.annotate.get_reference_gene_ids
    hlathena.annotate.add_tcga_expression


Prediction Tools
---------------

.. autosummary::
    :toctree: .
    
    hlathena.predict.predict


Amino Acid Featurization Tools
---------------

.. autosummary::
    :toctree: .

    hlathena.amino_acid_feature_map.AminoAcidFeatureMap
