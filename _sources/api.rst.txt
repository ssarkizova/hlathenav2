.. automodule:: hlathena
    :noindex:
    
API
===

*hlathena* can be used as a python package. Import hlathena by::

    import hlathena as hat
   
Analysis Tools
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
    hlathena.peptide_projection.PCA_numpy_SVD
    hlathena.peptide_projection.pep_pos_weight


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
    
Training Tools
---------------

.. autosummary::
    :toctree: .
    
    hlathena.trainer.run_training
    hlathena.trainer.trainer
    hlathena.trainer.PeptideNN
    hlathena.peptide_nn.PeptideRandomSampler
    hlathena.peptide_nn.train
    hlathena.peptide_nn.save_model
    
    
Model Evaluation Tools
---------------

.. autosummary::
    :toctree: .
    
Amino Acid Featurization Tools
---------------

.. autosummary::
    :toctree: .
        hlathena.amino_acid_feature_map.AminoAcidFeatureMap
