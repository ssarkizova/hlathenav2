from typing import List, Dict
from hlathena.peptide_dataset_train import PeptideDatasetTrain
from hlathena import peptide_nn
from hlathena.training_evaluation import TrainingEvaluation

import torch.utils.data as torch_data
import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import datetime
import random
import glob
import argparse
import logging


def create_config_dict(device, epochs, lr, dr, batch_size, decoy_mul, eval_decoy_ratio, fold, aa_features, featname,
                       pepfeats_dict, seed):
    return {
        'device': str(device),
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'dropout_rate': dr,
        'decoy_mul': decoy_mul,
        'eval_decoy_ratio': eval_decoy_ratio,
        'fold #': fold,
        'aa_feature_files': aa_features,
        'feature_set_name': featname,
        'feature_set_cols': pepfeats_dict[featname],
        'all_feature_sets': pepfeats_dict,
        'seed': seed
    }


def make_subdir(output_dir, subdir):
    subdir_path = os.path.join(output_dir, subdir)
    os.makedirs(subdir_path)
    return subdir_path


def make_output_dirs(output_dir, out_path):
    dir_name = '-'.join([get_currtime(), out_path])
    model_path = os.path.join(output_dir, dir_name)
    os.makedirs(model_path)

    models_dir = make_subdir(model_path, 'models')
    configs_dir = make_subdir(model_path, 'configs')
    eval_dir = make_subdir(model_path, 'eval')
    return models_dir, configs_dir, eval_dir


def get_currtime():
    return str(datetime.datetime.strftime(datetime.datetime.now(), format='%m.%d.%Y-%I.%M.%S%p'))


def get_aa_feature_files_from_dir(aa_dir):
    # setting the path for joining multiple files
    if (not os.path.isdir(aa_dir)):
        return []
    elif not len(os.listdir(aa_dir)):
        logging.warning("Amino acid feature directory empty.")
        return []

    featurefiles = os.path.join(aa_dir, "*.txt")

    # list of merged files returned
    return glob.glob(featurefiles)


def parse_feature_sets(set_file):
    with open(set_file, 'r') as f:
        featuresets = f.read().splitlines()
        featureset_dict = {}
        for s in featuresets:
            name, cols = s.split(":")
            cols = cols.split(",") if cols else []
            featureset_dict[name] = cols
    return featureset_dict


def get_dedup_pep_df(df):
    pep_grp = df.groupby("features")
    results = []
    index = []
    for p, d in pep_grp:
        r = [p] + list(d.mean(axis=0, numeric_only=True)) + list(d.std(axis=0, numeric_only=True))
        results.append(r)
        index.append(p)
    return pd.DataFrame(results,
                        columns=["features", "mean_auc", "mean_prauc", "mean_ppv", "std_auc", "std_prauc", "std_ppv"],
                        index=index)


def build_parser():
    parser = argparse.ArgumentParser(description='Run HLAthena training')

    parser.add_argument("-t", "--train_file",
                        help="peptide HLA train data file",
                        required=True,
                        type=str,
                        action='store')
    parser.add_argument("-v", "--val_file",
                        help="peptide HLA validation data file",
                        required=False,
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument("-del", "--delimiter",
                        help="training input file delimiter",
                        required=False,
                        type=str,
                        default=',',
                        action='store')
    parser.add_argument("-pc", "--pep_col",
                        help="specify peptide sequence column name for hits and decoys",
                        type=str,
                        default='pep',
                        action='store')
    parser.add_argument("-ac", "--allele_col",
                        help="allele column name",
                        type=str,
                        default="",
                        action='store')
    parser.add_argument("-tc", "--target_col",
                        help="target column name",
                        type=str,
                        action='store')
    parser.add_argument("--fold_col",
                        help="fold column name",
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument("-f", "--folds",
                        help="specify number of cross folds",
                        type=int,
                        default=5,
                        action='store')
    parser.add_argument("-e", "--epochs",
                        type=int,
                        default=10,
                        action="store")
    parser.add_argument("-lr", "--learning_rate",
                        type=float,
                        default=0.001,
                        action="store")
    parser.add_argument("-d", "--dropout_rate",
                        type=float,
                        default=0.1,
                        action="store")
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=32,
                        action="store")
    parser.add_argument("-pr", "--pred_replicates",
                        help="number of predictions per peptide for evaluation",
                        type=int,
                        default=100,
                        action='store')
    parser.add_argument("-dm", "--decoy_mul",
                        help="training ratio of hits to decoys e.g. 2.0 means 1:2 ratio of hits:decoys",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("-dr", "--decoy_ratio",
                        help="testing ratio of hits to decoys e.g. 2.0 means 1:2 ratio of hits:decoys",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("-rh", "--resampling_hits",
                        action='store_true')
    parser.add_argument("-af", "--assign_folds",
                        type=bool,
                        default=True)
    parser.add_argument("--aa_feature_folder",
                        help="comma-separated list of amino acid feature matrix files",
                        type=str,
                        default="",
                        action='store')
    parser.add_argument("--hla_encoding_file",
                        help="comma-separated table of hla features for training",
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument("-fc", "--feat_cols",
                        help="peptide-level feature columns to train on e.g. 'exp;clev'",
                        type=str,
                        default="",
                        action='store')
    parser.add_argument("-fs", "--feat_sets",
                        help="path to file containing feature combinations to run, one per row e.g. NAME:feat1,feat2",
                        type=str,
                        default="",
                        action='store')
    parser.add_argument("-r", "--repetitions",
                        help="number of times to repeat training",
                        type=int,
                        default=1,
                        action='store')
    parser.add_argument("-s", "--seeds",
                        help="optionally provide seeds for the training reps as comma-delimited list",
                        type=str,
                        default=None,
                        action='store')
    parser.add_argument("-o", "--outdir",
                        help="training output folder",
                        type=str,
                        default="",
                        action='store')
    parser.add_argument("-n", "--run_name",
                        help="run identifier",
                        type=str,
                        default="",
                        action='store')

    return parser.parse_args()


def check_input_args(args):
    logging.info('')
    logging.info('Checking input arguments')
    # check required arguments
    if args.train_file is None:
        raise Exception('provide training data')
    elif not os.path.exists(args.train_file):
        raise FileNotFoundError(f'training data not found: {args.train_file}')
    else:
        logging.info(f'train_file={args.train_file}')

    if args.val_file is None:
        logging.info('no validation data provided')
    elif not os.path.exists(args.val_file):
        raise FileNotFoundError(f'validation data not found: {args.val_file}')
    else:
        logging.info(f'val_file={args.val_file}')


    logging.info(f'delimiter={args.delimiter}')
    logging.info(f'pep_col={args.pep_col}')
    logging.info(f'allele_col={args.allele_col}')
    logging.info(f'target_col={args.target_col}')
    logging.info(f'fold_col={args.fold_col}')
    logging.info(f'folds={args.folds}')
    logging.info(f'epochs={args.epochs}')
    logging.info(f'learning_rate={args.learning_rate}')
    logging.info(f'dropout_rate={args.dropout_rate}')
    logging.info(f'batch_size={args.batch_size}')
    logging.info(f'pred_replicates={args.pred_replicates}')
    logging.info(f'repetitions={args.repetitions}')
    logging.info(f'decoy_mul={args.decoy_mul}')
    logging.info(f'decoy_ratio={args.decoy_ratio}')
    logging.info(f'resampling_hits={args.resampling_hits}')

    if args.aa_feature_folder is None or args.aa_feature_folder == '':
        logging.info(f'no aa_feature_folder provided')
    elif not os.path.exists(args.aa_feature_folder):
        raise FileNotFoundError(f'aa_feature_folder not found: {args.aa_feature_folder}')
    else:
        logging.info(f'aa_feature_folder={args.aa_feature_folder}')

    logging.info(f'outdir={args.outdir}')
    logging.info(f'assign_folds={args.assign_folds}')
    logging.info(f'run_name={args.run_name}')
    logging.info(f'seeds={args.seeds}')

    if args.hla_encoding_file is None:
        logging.info('no hla_encoding_file provided, using default')
    elif not os.path.exists(args.hla_encoding_file):
        raise FileNotFoundError(f'hla_encoding_file not found: {args.hla_encoding_file}')
    else:
        logging.info(f'hla_encoding_file={args.hla_encoding_file}')

    logging.info(f'feat_cols={args.feat_cols}')

    if args.feat_sets is None:
        logging.info('no feature set file, using feat_cols if provided')
    elif not os.path.exists(args.feat_sets):
        raise FileNotFoundError(f'feature set file not found: {args.feat_sets}')
    else:
        logging.info(f'feat_sets={args.feat_sets}')


def main():
    args = build_parser()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    log_file_path = os.path.join(args.outdir, f'{args.run_name}.{get_currtime()}.training.log')
    logging.basicConfig(filename=log_file_path,
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S%p')
    logging.info('Logging to: {}'.format(log_file_path))

    check_input_args(args)

    train_file = args.train_file
    val_file = args.val_file
    delimiter = args.delimiter
    pep_col = args.pep_col
    allele_col = args.allele_col
    tgt_col = args.target_col
    fold_col = args.fold_col
    folds = args.folds
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout_rate = args.dropout_rate
    batch_size = args.batch_size
    pred_replicates = args.pred_replicates
    reps = args.repetitions
    decoy_mul = args.decoy_mul
    decoy_ratio = args.decoy_ratio
    resampling_hits = args.resampling_hits
    aa_feature_folder = args.aa_feature_folder
    outdir = args.outdir
    assign_folds = args.assign_folds
    run_name = args.run_name
    seeds = args.seeds
    hla_encoding_file = args.hla_encoding_file

    pep_feature_cols = [] if len(args.feat_cols) == 0 else args.feat_cols.split(";")
    pep_feature_sets_dict = {str(pep_feature_cols): [col for col in pep_feature_cols]} if not args.feat_sets \
        else parse_feature_sets(args.feat_sets)
    # override peptide column input if feature set dict is provided
    pep_feature_cols = list({x for f in list(pep_feature_sets_dict.values()) if f for x in f}) if args.feat_sets \
        else pep_feature_cols
    pep_feature_cols.append(pep_col)
    seeds = None if seeds is None else seeds.split(';')

    # retrieve list of amino acid feature files
    aa_feature_files = get_aa_feature_files_from_dir(aa_feature_folder)

    train_df = pd.read_csv(train_file, sep=delimiter)

    assert (all([np.issubdtype(train_df[c], np.number) for c in pep_feature_cols if c != pep_col]))

    rep_metrics = []

    logging.info('Loading training dataset...')
    peptide_dataset = PeptideDatasetTrain(train_df,
                                          pep_col_name=pep_col,
                                          allele_col_name=allele_col,
                                          target_col_name=tgt_col,
                                          fold_col_name=fold_col,
                                          folds=folds,
                                          aa_feature_files=aa_feature_files,
                                          hla_encoding_file=hla_encoding_file)

    if val_file is not None:
        logging.info('Loading validation dataset...')
        val_df = pd.read_csv(val_file, sep=delimiter)
        val_dataset = PeptideDatasetTrain(val_df,
                                          pep_col_name=pep_col,
                                          allele_col_name=allele_col,
                                          target_col_name=tgt_col,
                                          fold_col_name=fold_col,
                                          folds=folds,
                                          aa_feature_files=aa_feature_files,
                                          hla_encoding_file=hla_encoding_file)
    else:
        logging.warning('No validation dataset provided, skipping validation...')
        val_dataset = None

    for rep in range(1, reps + 1):
        logging.info('')
        logging.info(f'Training rep {rep} of {reps} reps...')

        rep_outdir = os.path.join(outdir, f'rep{str(rep)}')
        seed = random.randrange(0, 1000) if seeds is None else int(seeds[rep - 1])
        logging.info(f"Rep {rep} using seed = {seed}")

        metrics = trainer(peptide_dataset=peptide_dataset,
                          folds=folds,
                          epochs=epochs,
                          learning_rate=learning_rate,
                          dropout_rate=dropout_rate,
                          batch_size=batch_size,
                          pred_replicates=pred_replicates,
                          decoy_mul=decoy_mul,
                          decoy_ratio=decoy_ratio,
                          resampling_hits=resampling_hits,
                          aa_feature_files=aa_feature_files,
                          output_dir=rep_outdir,
                          featsets_dict=pep_feature_sets_dict,
                          run_name=run_name,
                          seed=seed,
                          reassign_folds=assign_folds,
                          val_dataset=val_dataset)

        rep_metrics.extend(metrics)

        logging.info(f"Training finished for rep {rep}. Outputs stored in {rep_outdir}")

    logging.info(f'Writing all_metrics file to {os.path.join(outdir, "all_metrics.tsv")}')
    all_metrics = pd.DataFrame(rep_metrics, columns=["features", "auc", "prauc", "ppv"])
    all_metrics.to_csv(os.path.join(outdir, 'all_metrics.tsv'), sep='\t', index=False)

    # create summary plots if running multiple reps
    if reps > 1:
        logging.info('')
        logging.info(f'Creating summary evaluation tables and figures across {reps} replicates')
        logging.info(f'Writing summary metric table to {os.path.join(outdir, "summary_metrics.tsv")}')
        summ_metrics = get_dedup_pep_df(all_metrics)
        summ_metrics.to_csv(os.path.join(outdir, 'summary_metrics.tsv'), sep='\t', index=False)

        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_prauc", "std_prauc", outdir,
                                                         "prauc_barplot.png")
        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_ppv", "std_ppv", outdir, "ppv_barplot.png")

def trainer(peptide_dataset: PeptideDatasetTrain,
            folds: int,
            epochs: int,
            learning_rate: float,
            dropout_rate: float,
            batch_size: int,
            pred_replicates: int,
            decoy_mul: int,
            decoy_ratio: int,
            resampling_hits: bool,
            aa_feature_files: List[os.PathLike],
            output_dir: os.PathLike,
            featsets_dict: Dict[str, List[str]] = {},
            run_name: str = "", seed: int = None,
            reassign_folds: bool = True,
            val_dataset: PeptideDatasetTrain = None):


    metric_rows = []

    if reassign_folds:
        peptide_dataset.reassign_folds(folds, seed=seed)

    for featname in featsets_dict:
        logging.info(f"Training feature set {str(featname)} of {str(list(featsets_dict.keys()))}")

        feat_set = featsets_dict[featname]
        run_out_path = '-'.join([run_name, featname]) if run_name else featname

        models_dir, configs_dir, eval_dir = make_output_dirs(output_dir=output_dir, out_path=run_out_path)

        eval_dict = {"peptide": [], "allele": [], "target": [], "pred_mean": [], "pred_var": [], "fold": []}
        cross_fold_dict = {"fold": [], "ppv": [], "prauc": []}

        peptide_dataset.set_feat_cols(feat_set)
        # feature_dims = peptide_dataset.feature_dimensions()

        # if val_dataset is not None:
        #     val_dataset.set_feat_cols(feat_set)
        #     assert (feature_dims == val_dataset.feature_dimensions())

        for fold in range(folds):

            logging.info(f"Training fold {str(fold + 1)} of {folds}")
            logging.info(f"Decoy mul: {decoy_mul}")
            logging.info(f"Decoy ratio: {decoy_ratio}")
            logging.info(f"Resampling hits: {resampling_hits}")

            train_ids = peptide_dataset.get_train_idxs(fold=fold, decoy_mul=decoy_mul, resampling_hits=resampling_hits,
                                                       seed=seed)
            test_ids = peptide_dataset.get_test_idxs(fold=fold, decoy_ratio=decoy_ratio, seed=seed)

            unique, counts = np.unique(peptide_dataset.pep_df['ha__target'][train_ids], return_counts=True)
            logging.info(f"""Train split:
                            \n{np.asarray((unique, counts))}
                         """)

            unique, counts = np.unique(peptide_dataset.pep_df['ha__target'][test_ids], return_counts=True)
            logging.info(f"""Test split:
                            \n{np.asarray((unique, counts))}
                         """)

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = peptide_nn.PeptideRandomSampler(train_ids, seed)
            test_subsampler = peptide_nn.PeptideRandomSampler(test_ids, seed)

            # Define data loaders for training and testing data in this fold
            trainloader = torch_data.DataLoader(peptide_dataset, batch_size=batch_size, sampler=train_subsampler)
            testloader = torch_data.DataLoader(peptide_dataset, batch_size=batch_size, sampler=test_subsampler)

            try:
                val_subsampler = peptide_nn.PeptideRandomSampler([i for i in range(len(val_dataset))], seed)
                valloader = torch_data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_subsampler)
            except TypeError:
                valloader = None

            if torch.cuda.is_available():
                device = torch.device("cuda")
                devices = [x for x in range(torch.cuda.device_count())]
            else:
                device = torch.device("cpu")
                devices = []

            model = BigMHC(mhclen=peptide_dataset.PepHLAEncoder.hla_feature_dim)
            model = BigMHC.accelerate(
                model=model,
                devices=devices).train()

            # if torch.cuda.device_count() > 1:
            #     logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            #     model = nn.DataParallel(model)
            # model.to(device)
            # logging.info(f"Device: {str(device)}")
            optimizer_dict = train(model, trainloader, learning_rate, epochs)#, device, valloader)

            # Create model and train
            model_config = create_config_dict(device=device, epochs=epochs, lr=learning_rate, dr=dropout_rate,
                                              batch_size=batch_size, decoy_mul=decoy_mul, eval_decoy_ratio=decoy_ratio,
                                              fold=fold, aa_features=aa_feature_files,
                                              featname=featname, pepfeats_dict=featsets_dict, seed=seed)

            peptide_nn.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)

            targets, indices, preds = peptide_nn.evaluate(model, testloader, pred_replicates, device)
            targets = [t.item() for t in torch.hstack(targets).cpu()]
            indices = [i.item() for i in torch.hstack(indices).cpu()]
            preds = torch.vstack(preds).cpu()

            input_peps, input_alleles = [peptide_dataset.pep_at(i) for i in indices], [peptide_dataset.allele_at(i) for
                                                                                       i in
                                                                                       indices]
            pred_means = preds.mean(dim=-1).numpy()
            pred_vars = preds.var(dim=-1).numpy()

            for p, a, t, m, v in zip(input_peps, input_alleles, targets, pred_means, pred_vars):
                eval_dict['peptide'].append(p)
                eval_dict['allele'].append(a)
                eval_dict['target'].append(t)
                eval_dict['pred_mean'].append(m)
                eval_dict['pred_var'].append(v)
                eval_dict['fold'].append(str(fold + 1))

        eval_df = pd.DataFrame.from_dict(eval_dict)

        ## getting cross-fold metrics
        for fold, df in eval_df.groupby("fold"):
            train_eval = TrainingEvaluation(eval_df=df, decoy_ratio=True)
            cross_fold_dict['fold'].append(fold)
            cross_fold_dict['ppv'].append(train_eval.get_ppv())
            cross_fold_dict['prauc'].append(train_eval.get_prauc())

        crossfold_df = pd.DataFrame.from_dict(cross_fold_dict)
        crossfold_df.to_csv(os.path.join(eval_dir, "crossfold_eval.tsv"), index=False, sep='\t')

        eval_path_name = '{}_{}_eval.txt'.format(run_out_path, get_currtime())
        eval_df.to_csv(os.path.join(eval_dir, eval_path_name), index=False, sep='\t')

        train_eval = TrainingEvaluation(eval_df=eval_df, seed=seed, decoy_ratio=True)
        auc, prauc, ppv = train_eval.get_auc(), train_eval.get_prauc(), train_eval.get_ppv()

        logging.info(
            f"""Training and evaluation finished for {featname} {os.path.basename(output_dir)}.
                 AUC: {auc}
                 PPV: {ppv}
                 PRAUC: {prauc}
            """)

        metric_rows.append([featname, auc, prauc, ppv])

    eval_metrics = pd.DataFrame(metric_rows, columns=['features', 'auc', 'prauc', 'ppv'])
    eval_metrics.to_csv(os.path.join(output_dir, 'eval_metrics.tsv'), sep='\t', index=False)
    return metric_rows


from collections import OrderedDict


class AttentionLSTMCell(torch.nn.Module):

    def __init__(self, inp, out):
        super().__init__()
        self.att = torch.nn.MultiheadAttention(
            embed_dim=inp,
            num_heads=1,
            batch_first=True)
        self.lstm = torch.nn.LSTM(
            input_size=inp,
            hidden_size=out,
            batch_first=True,
            bidirectional=True)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.att(
            query=x,
            key=x,
            value=x,
            need_weights=False)
        y, _ = self.lstm(x)
        return y


class Dense(torch.nn.Module):

    def __init__(self, inp, out, act, drp):
        super().__init__()
        self.linear = torch.nn.Linear(inp, out)
        self.dropout = torch.nn.Dropout(drp)
        self.act = act

    def forward(self, x):
        return torch.cat((x, self.dropout(self.act(self.linear(x)))), -1)


class DenseBlock(torch.nn.Module):
    def __init__(self, inp, out, act, drp=0.5, layers=1):
        super().__init__()
        self.denseLayers = torch.nn.ModuleList(
            [Dense(
                inp=inp + (x * out),
                out=out,
                act=act,
                drp=drp) for x in range(layers)])

    def forward(self, x):
        for dense in self.denseLayers:
            x = dense(x)
        return x


class BigMHC(torch.nn.Module):
    class AnchorBlock(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, layers):
            super().__init__()
            self.enclen = enclen
            self.minpep = minpep
            self.denseBlock = DenseBlock(
                inp=mhclen + (minpep * enclen),
                out=hidlen,
                act=torch.nn.Tanh(),
                layers=layers)

        def buildInput(self, mhc, pep):
            return torch.cat(
                (mhc,
                 pep[:, :(self.minpep // 2) * self.enclen],
                 pep[:, -(self.minpep - (self.minpep // 2)) * self.enclen:]), 1)

        def forward(self, mhc, pep):
            return self.denseBlock(self.buildInput(mhc, pep))

    class PMHCLSTM(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, ncells):
            super().__init__()
            self.minpep = minpep
            self.enclen = enclen
            self.cells = torch.nn.ModuleList(
                [AttentionLSTMCell(
                    inp=2 * hidlen * x + mhclen + (minpep * enclen),
                    out=hidlen) for x in range(ncells)])

        def buildInput(self, mhc, pep, win):
            window = win * self.enclen
            slices = (pep.shape[1] - window) // self.enclen + 1
            output = torch.zeros(
                size=(mhc.shape[0], slices, mhc.shape[1] + window),
                dtype=torch.float32,
                device=mhc.device)
            output[:, :, window:] = torch.unsqueeze(mhc, 1)
            for i in range(slices):
                output[:, i, :window] = pep[:, i * self.enclen:i * self.enclen + window]
            return output

        def forward(self, mhc, pep):
            inp = self.buildInput(mhc, pep, self.minpep)
            for idx in range(len(self.cells)):
                out = self.cells[idx](inp)
                if idx < len(self.cells) - 1:
                    inp = torch.cat((inp, out), -1)
            return out[:, -1, :]

    class Condenser(torch.nn.Module):
        def __init__(self, mhclen, minpep, enclen, hidlen, layers):
            super().__init__()
            self.enclen = enclen
            self.act = torch.nn.Tanh()
            inp = mhclen + (minpep * enclen) + hidlen * (2 + layers)
            self.preAttentionDenseBlock = DenseBlock(
                inp=inp,
                out=hidlen,
                act=self.act,
                layers=layers)
            self.att = torch.nn.Linear(inp + hidlen * layers, mhclen)
            self.out = torch.nn.Linear(mhclen, 1)

        def forward(self, mhc, anchorOutput, lstmOutput):
            attention = self.att(self.preAttentionDenseBlock(
                torch.cat((anchorOutput, lstmOutput), -1)))
            attention = (torch.masked_select(
                self.out.weight * attention,
                mhc.bool()))
            attention = attention.reshape(mhc.shape[0], -1)
            return \
                (torch.sum(attention, dim=1) + self.out.bias,
                 attention)

    def __init__(
            self,
            mhclen=414,
            minpep=8,
            enclen=20,
            hidlen=1024,
            layers=2):

        super().__init__()

        self.lstm = BigMHC.PMHCLSTM(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            ncells=layers)
        self.anchorBlock = BigMHC.AnchorBlock(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            layers=layers)
        self.condenser = BigMHC.Condenser(
            mhclen=mhclen,
            minpep=minpep,
            enclen=enclen,
            hidlen=hidlen,
            layers=layers)

    def forward(self, mhc, pep):
        mhc = mhc.float()
        pep = pep.float()
        return self.condenser(
            mhc=mhc,
            anchorOutput=self.anchorBlock(mhc, pep),
            lstmOutput=self.lstm(mhc, pep))

    @staticmethod
    def accelerate(model, devices):
        """
        Based on devices arg, model is sent to either the CPU, a single GPU,
        or multiple GPUs using Torch DataParallel. If using DataParallel,
        the model is first pushed to the first GPU in the devices list.
        """
        if isinstance(model, torch.nn.parallel.DataParallel):
            model = model.module
        if not len(devices):
            return model.cpu()
        if len(devices) > 1:
            model = torch.nn.parallel.DataParallel(model, device_ids=devices)
        return model.to(devices[0])

    @staticmethod
    def decelerate(model):
        """
        Sends model to the CPU and returns the resulting model
        """
        if isinstance(model, torch.nn.parallel.DataParallel):
            return model.module.cpu()
        return model.cpu()

    @staticmethod
    def tllayers():
        return [
            "condenser.att.weight",
            "condenser.att.bias",
            "condenser.out.weight",
            "condenser.out.bias"]


# def train(model, trainloader, device):
def train(model, trainloader, learning_rate, epochs):  # , valloader = None, patience = 5, min_delta = 0):
    # logging.info("starting training on devices: {}".format(device))
    #
    # model = BigMHC.accelerate(
    #     model=model,
    #     devices=device).train()

    dev = next(model.parameters()).device
    lossf = torch.nn.BCEWithLogitsLoss().to(dev)

    optmz = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate)

    for ep in range(epochs):
        eperr = 0
        # data.dataset.makebats(
        #     maxbat=args.maxbat,
        #     shuffle=True,
        #     negfrac=None if args.transferlearn else 0.99)
        for _, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            input_pep = data[0].to(torch.float).to(dev)
            input_hla = data[1].to(torch.float).to(dev)
            labels = data[2].to(dev)

            # tgt = bat.tgt.float().to(dev)

            optmz.zero_grad()
            out, _ = model(
                mhc=input_hla,
                pep=input_pep)
            err = lossf(out, labels)
            err.backward()
            eperr += float(err) / len(data)
            optmz.step()

        logging.info("ep {} loss: {}".format(ep + 1, eperr))

    return optmz
    # if args.out:
    #     epdir = os.path.join(args.out, "ep{}".format(ep + 1))
    #     model = BigMHC.decelerate(model)
    #     model.save(epdir, tl=args.transferlearn)
    #     model = BigMHC.accelerate(model, args.devices)


def evaluate(model, dataloader, replicates, device):  # TODO: optional replicates, no dropout/model.train() if no rep
    """ Uses model to generate prediction with replicates for variance

    Args:
        model (PeptideNN): trained binding prediction model
        dataloader (DataLoader): peptide data
        replicates (int): number of replicates for prediction
        device (torch.device): device on which torch.Tensor will be allocated

    Returns:
        List of input peptides, list of target values, and list of predicted values

    """
    model.train()  # Set Training mode to enable dropouts
    with torch.no_grad():  # Not doing backward pass, just return predictions w/ dropout
        input_lst, target_lst, index_lst, prediction_lst = [], [], [], []

        # Iterate over the test data and generate predictions
        for _, data in enumerate(dataloader, 0):  # add dataset label to get item tuple?
            input_pep = data[0].to(torch.float).to(device)
            input_hla = data[1].to(torch.float).to(device)
            labels = data[2].to(device)
            indices = data[3].to(device)

            # input_lst.append(inputs)
            target_lst.append(labels)
            index_lst.append(indices)

            # Iterate over replicates
            predictions = torch.zeros(input_pep.shape[0], replicates)
            for j in range(0, replicates):
                out, _ = model(
                    mhc=input_hla,
                    pep=input_pep)
                logits = out.data
                predictions[:, j] = logits.squeeze()

            prediction_lst.append(predictions)

    # Combine data from epochs and return
    return target_lst, index_lst, prediction_lst

if __name__ == "__main__":
    main()