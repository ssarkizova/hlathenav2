"""Module for training and evaluating models"""
import os
import datetime
import random
import pandas as pd
import argparse
import random
import glob
import numpy as np
from pprint import pprint
import torch.utils.data as torch_data
import torch
from sklearn.model_selection import KFold

from typing import List, Dict
from hlathena.peptide_dataset import PeptideDataset
from hlathena import peptide_nn
from hlathena.training_evaluation import TrainingEvaluation


def get_aa_feature_files_from_dir(aa_dir):
    # setting the path for joining multiple files
    if (not os.path.isdir(aa_dir)):
        return []
    elif not len(os.listdir(aa_dir)):
        print("Amino acid feature directory empty.")
        return [] 

    featurefiles = os.path.join(aa_dir, "*.txt")

    # list of merged files returned
    return glob.glob(featurefiles)

def parse_feature_sets(set_file):
    with open(set_file,'r') as f:
        featuresets = f.read().splitlines()
        featureset_dict = {}
        for s in featuresets:
            name, cols = s.split(":")
            cols = cols.split(",") if cols else []
            featureset_dict[name] = cols
    return featureset_dict

def get_dedup_pep_df(df):
    pep_grp = df.groupby("features")
    results=[]
    index=[]
    for p, d in pep_grp:
        r = [p] + list(d.mean(axis=0,numeric_only=True)) + list(d.std(axis=0,numeric_only=True))
        results.append(r)
        index.append(p)
    return pd.DataFrame(results, columns=["features", "mean_auc", "mean_prauc", "mean_ppv", "std_auc", "std_prauc", "std_ppv"],index=index)


def create_config_dict(allele, epochs, lr, dr, batch_size, decoy_mul, fold, aa_features, featname, pepfeats_dict, seed):
    return {
        'allele': allele,
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout_rate': dr,
        'decoy_mul': decoy_mul,
        'fold #': fold,
        'aa_feature_files': aa_features,
        'feature_set_name': featname,
        'feature_set_cols': pepfeats_dict[featname],
        'all_feature_sets': pepfeats_dict,
        'seed': seed
    }


def make_subdir(output_dir, subdir):
    subdir_path = os.path.join(output_dir,subdir)
    os.makedirs(subdir_path)
    return subdir_path


def make_output_dirs(output_dir, out_path):
    dir_name = '_'.join([get_currtime(),out_path])
    model_path = os.path.join(output_dir,dir_name)
    os.makedirs(model_path)
    
    models_dir = make_subdir(model_path, 'models')
    configs_dir = make_subdir(model_path, 'configs')
    eval_dir = make_subdir(model_path, 'eval')
    return models_dir, configs_dir, eval_dir
    

def get_currtime():
    return str(datetime.datetime.now()).replace(" ","_")


def trainer(pep_df: pd.DataFrame, label_col: str, allele: str, folds: int, epochs: int, learning_rate: float, 
            dropout_rate: float, batch_size: int, pred_replicates: int, decoy_mul: int, aa_feature_files: List[os.PathLike], 
            output_dir: os.PathLike, featsets_dict: Dict[str, List[str]]={}, run_name: str = "", seed: int = None):
    """
    Train and evaluate a peptide prediction model.

    Args:
        pep_df (pd.DataFrame): DataFrame containing peptide data.
        label_col (str): Name of the column containing the target labels.
        allele (str): Allele for which the model is trained.
        folds (int): Number of folds for cross-validation.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        dropout_rate (float): Dropout rate for the model.
        batch_size (int): Batch size for training.
        pred_replicates (int): Number of prediction replicates.
        decoy_mul (int): Decoy multiplier for negative training samples.
        aa_feature_files (List[os.PathLike]): List of file paths for amino acid feature files.
        output_dir (os.PathLike): Output directory for saving model files and evaluation results.
        featsets_dict (Dict[str, List[str]], optional): Dictionary mapping feature set names to feature column names. Defaults to {}.
        run_name (str, optional): Name for the current run. Defaults to "".
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        List[List]: List of lists containing evaluation metrics for each feature set.

    """
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    kfold = KFold(n_splits=folds, random_state=seed, shuffle=True)

    metric_rows = []

    for featname in featsets_dict:
        print("Training feature set {} of {}...\n".format(str(featname),str(list(featsets_dict.keys()))))

        feat_set = featsets_dict[featname]
        run_out_path = '_'.join([run_name,featname]) if run_name else featname

        peptide_dataset = PeptideDataset(pep_df, aa_featurefiles=aa_feature_files, label_col=label_col, feat_cols=feat_set)

        models_dir, configs_dir, eval_dir = make_output_dirs(output_dir=output_dir, out_path=run_out_path)
    
        feature_dims = peptide_dataset.feature_dimensions

        eval_dict = {"peptide":[], "target":[], "pred_mean":[], "pred_var":[], "fold":[]}

        for fold, (train_ids, test_ids) in enumerate(kfold.split(peptide_dataset)):
            upsample_hits = True # TODO: make this a parameter
            if upsample_hits:
                class_counts = pep_df.iloc[train_ids][label_col].value_counts()
                hit_mul = class_counts.iloc[0]//class_counts.iloc[1] # get rounded ratio of hits to decoys 

                resampled_train_ids = []
                for i in train_ids:
                    if pep_df.iloc[i][label_col] == 1:
                        resampled_train_ids += hit_mul * [i] # resample hits to create equal ratio w/in fold
                    else:
                        resampled_train_ids += [i]
                train_ids = resampled_train_ids
            
            print("Training fold {} of {}...".format(str(fold+1), folds))
            # print("Train IDs for fold {} of feat set {}: {}".format(str(fold+1),str(feat_set), str(train_ids)))

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = peptide_nn.PeptideRandomSampler(train_ids, seed)
            test_subsampler = peptide_nn.PeptideRandomSampler(test_ids, seed)

            # Define data loaders for training and testing data in this fold
            trainloader = torch_data.DataLoader(peptide_dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch_data.DataLoader(peptide_dataset, batch_size=batch_size, sampler=test_subsampler)

            # Create model and train
            model_config = create_config_dict(
                                allele=allele, epochs=epochs, lr=learning_rate, dr=dropout_rate, 
                                batch_size=batch_size, decoy_mul=decoy_mul, fold=fold, aa_features=aa_feature_files,
                                featname=featname, pepfeats_dict=featsets_dict, seed=seed)
            # print(model_config)

            model = peptide_nn.PeptideNN(feature_dims, dropout_rate).to(device)
            optimizer_dict = peptide_nn.train(model, trainloader, learning_rate, epochs, device)

            peptide_nn.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)

            inputs, targets, preds = peptide_nn.evaluate(model, testloader, pred_replicates, device)
            inputs = torch.vstack(inputs).cpu()
            targets = [t.item() for t in torch.hstack(targets).cpu()]
            preds = torch.vstack(preds).cpu()

            input_peps = [peptide_dataset.decode_peptide(p) for p in inputs]
            pred_means = preds.mean(dim=-1).numpy()
            pred_vars = preds.var(dim=-1).numpy()

            for p, t, m, v in zip(input_peps, targets, pred_means, pred_vars):
                eval_dict['peptide'].append(p)
                eval_dict['target'].append(t)
                eval_dict['pred_mean'].append(m)
                eval_dict['pred_var'].append(v)
                eval_dict['fold'].append(str(fold+1))

            
        eval_df = pd.DataFrame.from_dict(eval_dict)
        eval_path_name = '{}_{}_eval.txt'.format(run_out_path, get_currtime())
        eval_df.to_csv(os.path.join(eval_dir,eval_path_name), index=False, sep='\t')

        train_eval = TrainingEvaluation(eval_df=eval_df, seed=seed)
        auc, prauc, ppv = train_eval.get_auc(), train_eval.get_prauc(), train_eval.get_ppv()
        
        print('\nTraining and evaluation finished for {run}.\n \
                 AUC: {auc}\n \
                 PPV: {ppv}\n \
                 PRAUC: {prauc}\n \
            '.format(run=featname, auc=auc, ppv=ppv, prauc=prauc))
        
        metric_rows.append([featname, auc, prauc, ppv])
    
    eval_metrics = pd.DataFrame(metric_rows, columns=['features','auc','prauc', 'ppv'])
    eval_metrics.to_csv(os.path.join(output_dir,'eval_metrics.tsv'), sep='\t', index=False)
    return metric_rows
    


def run_training(hits: os.PathLike, 
                 decoys: os.PathLike, 
                 allele: str = None, 
                 pep_col: str = 'seq', 
                 folds: int = 5, 
                 epochs: int = 4000, 
                 learning_rate: float = 0.001, 
                 dropout_rate: float = 0.1, 
                 batch_size: int = 32, 
                 pred_replicates: int = 100, 
                 decoy_mul: float = 1.0, 
                 aa_feature_folder: os.PathLike = None, 
                 feature_columns: str = None, 
                 feature_sets: os.PathLike = None, 
                 reps: int = 1, 
                 rep_seeds: str = None, 
                 outdir: os.PathLike = None, 
                 run: str = ""):
    """
    Run the training process for a peptide prediction model.

    Args:
        hits (os.PathLike): Path to the hits file.
        decoys (os.PathLike): Path to the decoys file.
        allele (str, optional): Allele for which the model is trained. Defaults to None.
        pep_col (str, optional): Name of the column containing the peptide sequence. Defaults to 'seq'.
        folds (int, optional): Number of folds for cross-validation. Defaults to 5.
        epochs (int, optional): Number of training epochs. Defaults to 4000.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.1.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        pred_replicates (int, optional): Number of prediction replicates. Defaults to 100.
        decoy_mul (float, optional): Decoy multiplier for negative training samples. Defaults to 1.0.
        aa_feature_folder (os.PathLike, optional): Path to the folder containing amino acid feature files. Defaults to None.
        feature_columns (str, optional): Names of the columns containing peptide features. Defaults to None.
        feature_sets (os.PathLike, optional): Path to the feature sets file. Defaults to None.
        reps (int, optional): Number of repetitions for training. Defaults to 1.
        rep_seeds (str, optional): Random seeds for each repetition. Defaults to None.
        outdir (os.PathLike, optional): Output directory for saving model files and evaluation results. Defaults to None.
        run (str, optional): Name for the current run. Defaults to "".

    """
    pep_feature_cols = [] if feature_columns is None else featur_columns.split(";")
    
    # override peptide column input if feature set dict is provided
    pep_feature_sets_dict = {str(pep_feature_cols): [col for col in pep_feature_cols]} if not feature_sets \
                                                                                       else parse_feature_sets(feature_sets)
    pep_feature_cols = list({x for f in list(pep_feature_sets_dict.values()) if f for x in f}) if feature_sets \
                                                                                         else pep_feature_cols                                                                       
    pep_feature_cols.append(pep_col)
    print(f'reps: {reps}, seeds: {rep_seeds}')
    seeds = None if rep_seeds is None or len(rep_seeds.split(','))!=reps else rep_seeds.split(',') 
    
    # retrieve list of amino acid feature files
    aa_feature_files = get_aa_feature_files_from_dir(aa_feature_folder)

    # get hits and decoys' dataframes
    if allele: # TODO: fix this...
        hits_df = pd.read_table(hits, sep=" ")
        hits_df = hits_df.loc[hits_df['allele'] == allele]
        hits_df = hits_df[pep_feature_cols]
    else:
        hits_df = pd.read_table(hits, sep=" ", usecols=pep_feature_cols)


    N = int(hits_df.shape[0]*decoy_mul) # get number of decoys to include based on decoy_mul
    decoys_df = pd.read_table(decoys, sep=" ", usecols=pep_feature_cols)
    # decoys_resampled = decoys_df.sample(N, random_state=1) # Moving this to loop
    
    print("Number of hits: %d" % len(hits_df))
    print("Number of decoys: %d" % len(decoys_df))

    # label pep column 'seq'
    default_pep_col = 'seq'
    hits_df.rename(columns={pep_col: default_pep_col}, inplace=True)
    decoys_df.rename(columns={pep_col: default_pep_col}, inplace=True)



    # check that both pep files have the same feature columns, if any
    assert(all(hits_df.columns.sort_values() == decoys_df.columns.sort_values()))
    
    # check that all peptide level feature columns are numeric
    assert(all([np.issubdtype(hits_df[c], np.number) for c in hits_df.columns if c!=default_pep_col]))
    assert(all([np.issubdtype(decoys_df[c], np.number) for c in decoys_df.columns if c!=default_pep_col]))
    
    rep_metrics = []
    
    hits_df['target']=1
    decoys_df['target']=0
    
    for rep in range(1,reps+1):
        print("\nTraining rep {} of {} reps...".format(rep,reps))

        rep_outdir = ''.join([outdir,'/rep',str(rep)])
        seed = random.randrange(0,100) if seeds is None else int(seeds[rep-1])
        print(f"SEED: {seed}\n")
        
        decoys_resampled = decoys_df.sample(N, random_state=seed)
        peps_df = pd.concat([hits_df,decoys_resampled])
        
        metrics = trainer(peps_df,'target',allele, folds, epochs, learning_rate, dropout_rate, batch_size, pred_replicates, decoy_mul, aa_feature_files, rep_outdir, pep_feature_sets_dict, run, seed)
        rep_metrics.extend(metrics)

        print("Training finished for rep {}. Outputs stored in {}".format(rep,rep_outdir))
    
    all_metrics = pd.DataFrame(rep_metrics, columns=["features","auc","prauc","ppv"])
    all_metrics.to_csv(os.path.join(outdir,'all_metrics.tsv'),sep='\t',index=False)
    
    # create summary plots if running multiple reps
    if reps>1:
        
        summ_metrics = get_dedup_pep_df(all_metrics)
        summ_metrics.to_csv(os.path.join(outdir,'summary_metrics.tsv'),sep='\t',index=False)

        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_prauc", "std_prauc", outdir, "prauc_barplot.png")
        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_ppv", "std_ppv", outdir, "ppv_barplot.png")


if __name__=="__main__" :
    parser=argparse.ArgumentParser()
    parser.add_argument("--hits", help="specify hits dataset", type=str, action='store') 
    parser.add_argument("--decoys", help="specify decoys dataset", type=str, action='store')
    parser.add_argument("--pep_column", help="specify peptide sequence column name for hits and decoys", type=str, default='seq', action='store')
    parser.add_argument("-a", "--allele", type=str, default="", action='store')
    parser.add_argument("-kf", "--number_folds", help="specify number of cross folds", type=int, default=5, action='store')
    parser.add_argument("-e", "--epochs", type=int, default=4000, action="store")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, action="store")
    parser.add_argument("-dr", "--dropout_rate", type=float, default=0.1, action="store") #TODO: add default dr
    parser.add_argument("-b", "--batch_size", type=int, default=32, action="store")
    parser.add_argument("-pr", "--pred_replicates", type=int, default=100, action='store')
    parser.add_argument("-dm", "--decoy_mul", help="training ratio of hits to decoys e.g. 2.0 means 1:2 ratio of hits:decoys", type=float, default=1.0, action='store')
    parser.add_argument("--aa_feature_folder", help="comma-separated list of amino acid feature matrix files", type=str, default="", action='store')
    parser.add_argument("--feat_cols", help="peptide-level feature columns to train on e.g. 'exp;clev'. Must exist in hits and decoys input files", type=str, default="", action='store')
    parser.add_argument("--feat_sets", help="path to file containing feature combinations to run, one per row e.g. NAME:feat1,feat2", type=str, default="", action='store')
    parser.add_argument("--repetitions", help="number of times to repeat training", type=int, default=1, action='store')
    parser.add_argument("--seeds", help="optionally provide seeds for the training reps as comma-delimited list", type=str, default=None, action='store')
    parser.add_argument("-o", "--outdir", help="where to store output", type=str, default="", action='store')
    parser.add_argument("-r", "--run_name", help="used to name output sub directory", type=str, default="", action='store')

    
    args = parser.parse_args()
    hits = args.hits
    decoys = args.decoys
    pep_col = args.pep_column
    allele = args.allele
    folds = args.number_folds
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    pred_replicates = args.pred_replicates
    decoy_mul = args.decoy_mul
    aa_feature_folder = args.aa_feature_folder
    feature_cols = args.feat_cols
    feature_sets = args.feat_sets
    seeds = args.seeds
    reps = args.repetitions
    outdir = args.outdir
    run_name = args.run_name
    
    run_training(hits = hits, decoys = decoys, 
                 allele = allele, 
                 pep_col = pep_col, 
                 folds = folds, epochs = epochs, learning_rate = learning_rate, dropout_rate = dropout_rate, batch_size = batch_size, 
                 pred_replicates = pred_replicates, 
                 decoy_mul = decoy_mul, 
                 aa_feature_folder = aa_feature_folder, feature_columns = feature_cols, feature_sets = feature_sets, 
                 repetitions = reps, rep_seeds = seeds, 
                 outdir = outdir, run = run_name)


