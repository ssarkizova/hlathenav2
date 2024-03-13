Search.setIndex({"docnames": ["api", "generated/hlathena", "hlathena.amino_acid_feature_map.AminoAcidFeatureMap", "hlathena.annotate.add_tcga_expression", "hlathena.annotate.get_reference_gene_ids", "hlathena.annotate.list_tcga_expression_references", "hlathena.peptide_nn.PeptideRandomSampler", "hlathena.peptide_nn.save_model", "hlathena.peptide_nn.train", "hlathena.peptide_projection.PCA_encode", "hlathena.peptide_projection.PCA_numpy_SVD", "hlathena.peptide_projection.get_peptide_clustering", "hlathena.peptide_projection.get_umap_embedding", "hlathena.peptide_projection.pep_pos_weight", "hlathena.plotting.plot_clustered_umap", "hlathena.plotting.plot_length", "hlathena.plotting.plot_logo", "hlathena.plotting.plot_umap", "hlathena.predict.predict", "hlathena.trainer.run_training", "hlathena.trainer.trainer", "index", "usage"], "filenames": ["api.rst", "generated/hlathena.rst", "hlathena.amino_acid_feature_map.AminoAcidFeatureMap.rst", "hlathena.annotate.add_tcga_expression.rst", "hlathena.annotate.get_reference_gene_ids.rst", "hlathena.annotate.list_tcga_expression_references.rst", "hlathena.peptide_nn.PeptideRandomSampler.rst", "hlathena.peptide_nn.save_model.rst", "hlathena.peptide_nn.train.rst", "hlathena.peptide_projection.PCA_encode.rst", "hlathena.peptide_projection.PCA_numpy_SVD.rst", "hlathena.peptide_projection.get_peptide_clustering.rst", "hlathena.peptide_projection.get_umap_embedding.rst", "hlathena.peptide_projection.pep_pos_weight.rst", "hlathena.plotting.plot_clustered_umap.rst", "hlathena.plotting.plot_length.rst", "hlathena.plotting.plot_logo.rst", "hlathena.plotting.plot_umap.rst", "hlathena.predict.predict.rst", "hlathena.trainer.run_training.rst", "hlathena.trainer.trainer.rst", "index.rst", "usage.rst"], "titles": ["API", "hlathena", "hlathena.amino_acid_feature_map.AminoAcidFeatureMap", "hlathena.annotate.add_tcga_expression", "hlathena.annotate.get_reference_gene_ids", "hlathena.annotate.list_tcga_expression_references", "hlathena.peptide_nn.PeptideRandomSampler", "hlathena.peptide_nn.save_model", "hlathena.peptide_nn.train", "hlathena.peptide_projection.PCA_encode", "hlathena.peptide_projection.PCA_numpy_SVD", "hlathena.peptide_projection.get_peptide_clustering", "hlathena.peptide_projection.get_umap_embedding", "hlathena.peptide_projection.pep_pos_weight", "hlathena.plotting.plot_clustered_umap", "hlathena.plotting.plot_length", "hlathena.plotting.plot_logo", "hlathena.plotting.plot_umap", "hlathena.predict.predict", "hlathena.trainer.run_training", "hlathena.trainer.trainer", "Welcome to hlathena\u2019s documentation!", "Usage"], "terms": {"hlathena": [0, 22], "can": [0, 12, 21], "us": [0, 3, 4, 8, 9, 10, 11, 18, 21, 22], "python": 0, "packag": 0, "import": 0, "hat": 0, "class": [2, 6], "aa_feature_fil": [2, 20], "list": [2, 4, 5, 9, 10, 13, 16, 18, 20, 21], "pathlik": [2, 7, 9, 18, 19, 20], "none": [2, 3, 4, 8, 9, 12, 14, 15, 16, 17, 19, 20, 21], "creat": [2, 6, 12], "an": 2, "amino": [2, 4, 9, 13, 19, 20, 21], "acid": [2, 4, 9, 13, 19, 20, 21], "featur": [2, 9, 13, 14, 17, 19, 20, 21], "map": [2, 20], "from": 2, "properti": 2, "tabl": 2, "feature_fil": 2, "path": [2, 4, 7, 9, 14, 17, 18, 19, 20], "type": [2, 3, 5, 6, 18, 20], "o": [2, 7, 18, 19, 20], "feature_map": 2, "panda": [2, 12, 18], "datafram": [2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21], "feature_count": 2, "integ": 2, "count": [2, 7], "aa": 2, "int": [2, 6, 7, 8, 9, 11, 12, 16, 18, 19, 20], "__init__": [2, 6, 21], "init": 2, "level": 2, "includ": 2, "file": [2, 4, 19, 20], "format": 2, "One": 2, "row": 2, "per": [2, 9], "index": [2, 9, 21], "column": [2, 3, 4, 9, 11, 14, 15, 16, 17, 19, 20, 21], "one": [2, 9, 11], "colum": 2, "header": 2, "space": 2, "delimit": 2, "paramet": [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], "matrix": [2, 9, 10, 13], "method": [2, 4, 6], "peptide_dataset": 3, "peptidedataset": [3, 4, 9], "cancer_typ": 3, "str": [3, 4, 9, 14, 15, 16, 17, 18, 19, 20, 21], "hugo_col": 3, "hugo_symbol": 3, "pep_col_nam": [3, 4], "add": [3, 4], "tcga": [3, 5], "refer": [3, 4, 5], "express": [3, 5], "data": [3, 6, 8, 12, 20, 21], "A": [3, 4, 11, 12, 14, 15, 16, 17, 21], "peptid": [3, 4, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], "sequenc": [3, 4, 12, 13, 15, 16, 17, 19, 21], "associ": 3, "hugo": 3, "symbol": 3, "cancer": [3, 5], "abbrevi": [3, 5], "name": [3, 14, 15, 16, 17, 19, 20, 21], "specifi": 3, "input": [3, 10], "return": [3, 4, 5, 8, 9, 10, 11, 12, 13, 15, 18, 20, 21], "append": [3, 11], "select": 3, "pep_df": [4, 15, 16, 20, 21], "ref_fasta": 4, "add_context": 4, "bool": [4, 14, 17], "true": 4, "gene": 4, "id": 4, "given": [4, 14, 16, 17, 18], "fasta": 4, "thi": 4, "function": [4, 16, 21], "identifi": 4, "": [4, 7], "produc": 4, "each": [4, 8, 18, 19, 20], "inform": 4, "about": 4, "correspond": 4, "option": [4, 14, 17, 19, 20], "flank": 4, "The": [4, 11, 14, 15, 16, 17, 21], "default": [4, 9, 14, 15, 16, 17, 18, 19, 20, 21], "i": [4, 9, 14, 15, 16, 17, 19, 20, 21], "hg19": 4, "proteom": 4, "contain": [4, 18, 19, 20], "protein": 4, "If": [4, 9, 16], "output": [4, 7, 19, 20], "30": 4, "upstream": 4, "downstream": 4, "avail": 5, "dataset": [5, 6, 9], "descript": 5, "seed": [6, 12, 19, 20, 21], "custom": 6, "random": [6, 12, 19, 20], "sampler": 6, "sampl": [6, 11, 19, 20], "np": 6, "ndarrai": [6, 10], "number": [6, 8, 11, 18, 19, 20], "gener": 6, "model": [7, 8, 18, 19, 20, 21], "fold": [7, 19, 20], "models_dir": 7, "configs_dir": 7, "optim": [7, 8, 19, 20], "config": 7, "save": [7, 9, 14, 17, 18, 19, 20], "configur": 7, "info": 7, "directori": [7, 19, 20], "peptidenn": [7, 8], "train": [7, 19, 20, 21], "bind": [7, 8], "predict": [7, 8, 19, 20, 21], "dir": 7, "locat": 7, "subdir": 7, "adam": 7, "dict": [7, 20], "dictionari": [7, 20], "detail": 7, "trainload": 8, "learning_r": [8, 19, 20], "epoch": [8, 19, 20], "devic": 8, "valload": 8, "patienc": 8, "5": [8, 12, 19], "min_delta": 8, "0": [8, 12, 18, 19], "dataload": 8, "float": [8, 10, 12, 13, 18, 19, 20], "step": 8, "size": [8, 19, 20], "iter": 8, "torch": 8, "which": [8, 12, 14, 15, 16, 17, 19, 20, 21], "tensor": 8, "alloc": 8, "valid": [8, 19, 20], "set": [8, 12, 19, 20], "earli": 8, "stop": 8, "minimum": 8, "loss": 8, "differ": 8, "peplen": 9, "allel": [9, 13, 16, 19, 20], "aa_feature_map": [9, 13], "aminoacidfeaturemap": [9, 13], "precomp_pca_path": 9, "save_pca_path": 9, "encod": [9, 11, 12, 13], "perform": [9, 14, 15, 16, 17, 18, 21], "pca": [9, 10, 11, 12], "provid": [9, 15, 16, 21], "kidera": 9, "factor": 9, "length": [9, 15, 16], "subset": 9, "befor": 9, "hla": 9, "precomput": 9, "object": [9, 15, 21], "pc": 9, "posn_weights_df": 10, "tupl": [10, 14, 17], "comput": 10, "svd": 10, "eigenvalu": 10, "eigenvector": 10, "explain": 10, "varianc": 10, "umap_embed": 11, "ep": 11, "3": 11, "min_sampl": 11, "7": 11, "cluster": [11, 14, 17], "dbscan": 11, "algorithm": 11, "umap": [11, 12, 14, 17], "embed": [11, 12, 14, 17], "maximum": 11, "distanc": 11, "between": [11, 12], "two": 11, "consid": 11, "neighborhood": 11, "other": 11, "total": 11, "weight": [11, 13], "point": [11, 12], "core": 11, "feature_matrix": 12, "n_neighbor": 12, "min_dist": 12, "random_st": 12, "control": 12, "balanc": 12, "local": 12, "v": 12, "global": 12, "structur": 12, "how": 12, "tightli": 12, "ar": [12, 14, 16, 17], "pack": 12, "togeth": 12, "state": 12, "valu": [12, 14, 15, 16, 17, 21], "ensur": 12, "reproduc": [12, 20], "encoded_pep_df": 13, "pos_weight": 13, "posit": 13, "averag": 13, "specif": 13, "pan": 13, "entropi": 13, "umap_embedding_df": [14, 17], "label_col": [14, 15, 16, 17, 20, 21], "titl": [14, 17], "save_path": [14, 17], "countplot_log_scal": [14, 17], "fals": [14, 17], "ax": [14, 15, 16, 17, 21], "seq": [14, 15, 16, 17, 19, 21], "umap_1": [14, 17], "umap_2": [14, 17], "expect": [14, 17], "denot": [14, 15, 16, 17, 21], "categori": [14, 15, 16, 17, 21], "comparison": [14, 15, 16, 17, 21], "case": [14, 15, 16, 17, 21], "where": [14, 17], "figur": [14, 17], "pep_col": [15, 16, 19, 21], "ha__pep": [15, 16, 21], "distribut": [15, 21], "matplotlib": [15, 21], "rais": [15, 16, 18, 21], "indexerror": [15, 16, 18, 21], "No": [15, 18, 21], "logo": 16, "all": 16, "found": 16, "indic": 17, "whether": 17, "ha": 17, "model_path": 18, "dropout_r": [18, 19, 20], "1": [18, 19], "replic": [18, 19, 20], "score": 18, "pytorch": 18, "dropout": [18, 19, 20], "rate": [18, 19, 20], "dure": 18, "submit": 18, "hit": 19, "decoi": [19, 20], "4000": 19, "001": 19, "batch_siz": [19, 20], "32": 19, "pred_repl": [19, 20], "100": 19, "decoy_mul": [19, 20], "aa_feature_fold": 19, "feature_column": 19, "feature_set": 19, "rep": 19, "rep_se": 19, "outdir": 19, "run": [19, 20], "process": 19, "cross": [19, 20], "learn": [19, 20], "batch": [19, 20], "multipli": [19, 20], "neg": [19, 20], "folder": 19, "repetit": 19, "evalu": [19, 20, 21], "result": [19, 20], "current": [19, 20], "output_dir": 20, "featsets_dict": 20, "run_nam": 20, "pd": 20, "target": 20, "label": 20, "metric": 20, "modul": 21, "search": 21, "page": 21, "usag": 21, "instal": 21, "api": 21, "analysi": 21, "tool": 21, "plot_length": 21, "plot_logo": 21, "plot_umap": 21, "plot_clustered_umap": 21, "project": 21, "annot": 21, "trainer": 21, "run_train": 21, "peptide_nn": 21, "peptiderandomsampl": 21, "save_model": 21, "To": [21, 22], "you": 21, "first": 22, "pip": 22, "venv": 22}, "objects": {"": [[1, 0, 0, "-", "hlathena"]], "hlathena.amino_acid_feature_map": [[2, 1, 1, "", "AminoAcidFeatureMap"]], "hlathena.amino_acid_feature_map.AminoAcidFeatureMap": [[2, 2, 1, "", "__init__"], [2, 3, 1, "", "feature_count"], [2, 3, 1, "", "feature_files"], [2, 3, 1, "", "feature_map"]], "hlathena.annotate": [[3, 4, 1, "", "add_tcga_expression"], [4, 4, 1, "", "get_reference_gene_ids"], [5, 4, 1, "", "list_tcga_expression_references"]], "hlathena.peptide_nn": [[6, 1, 1, "", "PeptideRandomSampler"], [7, 4, 1, "", "save_model"], [8, 4, 1, "", "train"]], "hlathena.peptide_nn.PeptideRandomSampler": [[6, 2, 1, "", "__init__"], [6, 3, 1, "", "data"], [6, 3, 1, "", "seed"]], "hlathena.peptide_projection": [[9, 4, 1, "", "PCA_encode"], [10, 4, 1, "", "PCA_numpy_SVD"], [11, 4, 1, "", "get_peptide_clustering"], [12, 4, 1, "", "get_umap_embedding"], [13, 4, 1, "", "pep_pos_weight"]], "hlathena.plotting": [[14, 4, 1, "", "plot_clustered_umap"], [21, 4, 1, "", "plot_length"], [16, 4, 1, "", "plot_logo"], [17, 4, 1, "", "plot_umap"]], "hlathena.predict": [[18, 4, 1, "", "predict"]], "hlathena.trainer": [[19, 4, 1, "", "run_training"], [20, 4, 1, "", "trainer"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:attribute", "4": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "function", "Python function"]}, "titleterms": {"api": 0, "analysi": 0, "tool": 0, "peptid": 0, "project": 0, "annot": [0, 3, 4, 5], "predict": [0, 18], "train": [0, 8], "model": 0, "evalu": 0, "amino": 0, "acid": 0, "featur": 0, "hlathena": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], "amino_acid_feature_map": 2, "aminoacidfeaturemap": 2, "add_tcga_express": 3, "get_reference_gene_id": 4, "list_tcga_expression_refer": 5, "peptide_nn": [6, 7, 8], "peptiderandomsampl": 6, "save_model": 7, "peptide_project": [9, 10, 11, 12, 13], "pca_encod": 9, "pca_numpy_svd": 10, "get_peptide_clust": 11, "get_umap_embed": 12, "pep_pos_weight": 13, "plot": [14, 15, 16, 17, 21], "plot_clustered_umap": 14, "plot_length": 15, "plot_logo": 16, "plot_umap": 17, "trainer": [19, 20], "run_train": 19, "welcom": 21, "": 21, "document": 21, "indic": 21, "tabl": 21, "content": 21, "length": 21, "usag": 22, "instal": 22}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 58}, "alltitles": {"API": [[0, "api"]], "Analysis Tools": [[0, "analysis-tools"]], "Peptide Projection Tools": [[0, "peptide-projection-tools"]], "Annotation Tools": [[0, "annotation-tools"]], "Prediction Tools": [[0, "prediction-tools"]], "Training Tools": [[0, "training-tools"]], "Model Evaluation Tools": [[0, "model-evaluation-tools"]], "Amino Acid Featurization Tools": [[0, "amino-acid-featurization-tools"]], "hlathena": [[1, "module-hlathena"]], "hlathena.amino_acid_feature_map.AminoAcidFeatureMap": [[2, "hlathena-amino-acid-feature-map-aminoacidfeaturemap"]], "hlathena.annotate.add_tcga_expression": [[3, "hlathena-annotate-add-tcga-expression"]], "hlathena.annotate.get_reference_gene_ids": [[4, "hlathena-annotate-get-reference-gene-ids"]], "hlathena.annotate.list_tcga_expression_references": [[5, "hlathena-annotate-list-tcga-expression-references"]], "hlathena.peptide_nn.PeptideRandomSampler": [[6, "hlathena-peptide-nn-peptiderandomsampler"]], "hlathena.peptide_nn.save_model": [[7, "hlathena-peptide-nn-save-model"]], "hlathena.peptide_nn.train": [[8, "hlathena-peptide-nn-train"]], "hlathena.peptide_projection.PCA_encode": [[9, "hlathena-peptide-projection-pca-encode"]], "hlathena.peptide_projection.PCA_numpy_SVD": [[10, "hlathena-peptide-projection-pca-numpy-svd"]], "hlathena.peptide_projection.get_peptide_clustering": [[11, "hlathena-peptide-projection-get-peptide-clustering"]], "hlathena.peptide_projection.get_umap_embedding": [[12, "hlathena-peptide-projection-get-umap-embedding"]], "hlathena.peptide_projection.pep_pos_weight": [[13, "hlathena-peptide-projection-pep-pos-weight"]], "hlathena.plotting.plot_clustered_umap": [[14, "hlathena-plotting-plot-clustered-umap"]], "hlathena.plotting.plot_length": [[15, "hlathena-plotting-plot-length"]], "hlathena.plotting.plot_logo": [[16, "hlathena-plotting-plot-logo"]], "hlathena.plotting.plot_umap": [[17, "hlathena-plotting-plot-umap"]], "hlathena.predict.predict": [[18, "hlathena-predict-predict"]], "hlathena.trainer.run_training": [[19, "hlathena-trainer-run-training"]], "hlathena.trainer.trainer": [[20, "hlathena-trainer-trainer"]], "Welcome to hlathena\u2019s documentation!": [[21, "welcome-to-hlathena-s-documentation"]], "Indices and tables": [[21, "indices-and-tables"]], "Contents": [[21, "contents"]], "Plotting length": [[21, "plotting-length"]], "Usage": [[22, "usage"]], "Installation": [[22, "installation"]]}, "indexentries": {"hlathena": [[1, "module-hlathena"]], "module": [[1, "module-hlathena"]], "aminoacidfeaturemap (class in hlathena.amino_acid_feature_map)": [[2, "hlathena.amino_acid_feature_map.AminoAcidFeatureMap"]], "__init__() (hlathena.amino_acid_feature_map.aminoacidfeaturemap method)": [[2, "hlathena.amino_acid_feature_map.AminoAcidFeatureMap.__init__"]], "feature_count (hlathena.amino_acid_feature_map.aminoacidfeaturemap attribute)": [[2, "hlathena.amino_acid_feature_map.AminoAcidFeatureMap.feature_count"]], "feature_files (hlathena.amino_acid_feature_map.aminoacidfeaturemap attribute)": [[2, "hlathena.amino_acid_feature_map.AminoAcidFeatureMap.feature_files"]], "feature_map (hlathena.amino_acid_feature_map.aminoacidfeaturemap attribute)": [[2, "hlathena.amino_acid_feature_map.AminoAcidFeatureMap.feature_map"]], "add_tcga_expression() (in module hlathena.annotate)": [[3, "hlathena.annotate.add_tcga_expression"]], "get_reference_gene_ids() (in module hlathena.annotate)": [[4, "hlathena.annotate.get_reference_gene_ids"]], "list_tcga_expression_references() (in module hlathena.annotate)": [[5, "hlathena.annotate.list_tcga_expression_references"]], "peptiderandomsampler (class in hlathena.peptide_nn)": [[6, "hlathena.peptide_nn.PeptideRandomSampler"]], "__init__() (hlathena.peptide_nn.peptiderandomsampler method)": [[6, "hlathena.peptide_nn.PeptideRandomSampler.__init__"]], "data (hlathena.peptide_nn.peptiderandomsampler attribute)": [[6, "hlathena.peptide_nn.PeptideRandomSampler.data"]], "seed (hlathena.peptide_nn.peptiderandomsampler attribute)": [[6, "hlathena.peptide_nn.PeptideRandomSampler.seed"]], "save_model() (in module hlathena.peptide_nn)": [[7, "hlathena.peptide_nn.save_model"]], "train() (in module hlathena.peptide_nn)": [[8, "hlathena.peptide_nn.train"]], "pca_encode() (in module hlathena.peptide_projection)": [[9, "hlathena.peptide_projection.PCA_encode"]], "pca_numpy_svd() (in module hlathena.peptide_projection)": [[10, "hlathena.peptide_projection.PCA_numpy_SVD"]], "get_peptide_clustering() (in module hlathena.peptide_projection)": [[11, "hlathena.peptide_projection.get_peptide_clustering"]], "get_umap_embedding() (in module hlathena.peptide_projection)": [[12, "hlathena.peptide_projection.get_umap_embedding"]], "pep_pos_weight() (in module hlathena.peptide_projection)": [[13, "hlathena.peptide_projection.pep_pos_weight"]], "plot_clustered_umap() (in module hlathena.plotting)": [[14, "hlathena.plotting.plot_clustered_umap"]], "plot_length() (in module hlathena.plotting)": [[15, "hlathena.plotting.plot_length"], [21, "hlathena.plotting.plot_length"]], "plot_logo() (in module hlathena.plotting)": [[16, "hlathena.plotting.plot_logo"]], "plot_umap() (in module hlathena.plotting)": [[17, "hlathena.plotting.plot_umap"]], "predict() (in module hlathena.predict)": [[18, "hlathena.predict.predict"]], "run_training() (in module hlathena.trainer)": [[19, "hlathena.trainer.run_training"]], "trainer() (in module hlathena.trainer)": [[20, "hlathena.trainer.trainer"]]}})