from typing import List, Dict
from hlathena.peptide_dataset_train import PeptideDatasetTrain
from hlathena import peptide_nn
from hlathena import peptide_transformer
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

    logging.info('')


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

    pep_feature_cols = [] if len(args.feat_cols) == 0 else args.feat_cols.split(";")
    pep_feature_sets_dict = {str(pep_feature_cols): [col for col in pep_feature_cols]} if not args.feat_sets \
        else parse_feature_sets(args.feat_sets)
    # override peptide column input if feature set dict is provided
    pep_feature_cols = list({x for f in list(pep_feature_sets_dict.values()) if f for x in f}) if args.feat_sets \
        else pep_feature_cols
    pep_feature_cols.append(args.pep_col)
    seeds = None if args.seeds is None else args.seeds.split(';')

    # retrieve list of amino acid feature files
    aa_feature_files = get_aa_feature_files_from_dir(args.aa_feature_folder)

    train_df = pd.read_csv(args.train_file, sep=args.delimiter)

    assert (all([np.issubdtype(train_df[c], np.number) for c in pep_feature_cols if c != args.pep_col]))

    rep_metrics = []

    logging.info('Loading training dataset...')
    peptide_dataset = PeptideDatasetTrain(train_df,
                                          pep_col_name=args.pep_col,
                                          allele_col_name=args.allele_col,
                                          target_col_name=args.target_col,
                                          fold_col_name=args.fold_col,
                                          folds=args.folds,
                                          aa_feature_files=aa_feature_files,
                                          hla_encoding_file=args.hla_encoding_file)

    if args.val_file is not None:
        logging.info('Loading validation dataset...')
        val_df = pd.read_csv(args.val_file, sep=args.delimiter)
        val_dataset = PeptideDatasetTrain(val_df,
                                          pep_col_name=args.pep_col,
                                          allele_col_name=args.allele_col,
                                          target_col_name=args.target_col,
                                          fold_col_name=args.fold_col,
                                          folds=args.folds,
                                          aa_feature_files=aa_feature_files,
                                          hla_encoding_file=args.hla_encoding_file)
    else:
        logging.warning('No validation dataset provided, skipping validation...')
        val_dataset = None

    for rep in range(1, args.repetitions + 1):
        logging.info('')
        logging.info(f'Training rep {rep} of {args.repetitions} reps...')

        rep_outdir = os.path.join(args.outdir, f'rep{str(rep)}')
        seed = random.randrange(0, 1000) if seeds is None else int(seeds[rep - 1])
        logging.info(f"Rep {rep} using seed = {seed}")

        metrics = trainer(args,
                          peptide_dataset=peptide_dataset,
                          val_dataset=val_dataset,
                          output_dir=rep_outdir,
                          seed=seed)

        rep_metrics.extend(metrics)

        logging.info(f"Training finished for rep {rep}. Outputs stored in {rep_outdir}")

    logging.info(f'Writing all_metrics file to {os.path.join(args.outdir, "all_metrics.tsv")}')
    all_metrics = pd.DataFrame(rep_metrics, columns=["features", "auc", "prauc", "ppv"])
    all_metrics.to_csv(os.path.join(args.outdir, 'all_metrics.tsv'), sep='\t', index=False)

    # create summary plots if running multiple reps
    if args.repetitions > 1:
        logging.info('')
        logging.info(f'Creating summary evaluation tables and figures across {args.repetitions} replicates')
        logging.info(f'Writing summary metric table to {os.path.join(args.outdir, "summary_metrics.tsv")}')
        summ_metrics = get_dedup_pep_df(all_metrics)
        summ_metrics.to_csv(os.path.join(args.outdir, 'summary_metrics.tsv'), sep='\t', index=False)

        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_prauc", "std_prauc", args.outdir,
                                                         "prauc_barplot.png")
        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_ppv", "std_ppv", args.outdir, "ppv_barplot.png")


def train_preset_split():
    # Load arguments (modified from main())
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

    pep_feature_cols = [] if len(args.feat_cols) == 0 else args.feat_cols.split(";")
    pep_feature_sets_dict = {str(pep_feature_cols): [col for col in pep_feature_cols]} if not args.feat_sets \
        else parse_feature_sets(args.feat_sets)
    # override peptide column input if feature set dict is provided
    pep_feature_cols = list({x for f in list(pep_feature_sets_dict.values()) if f for x in f}) if args.feat_sets \
        else pep_feature_cols
    pep_feature_cols.append(args.pep_col)
    seeds = None if args.seeds is None else args.seeds.split(';')

    # retrieve list of amino acid feature files
    aa_feature_files = get_aa_feature_files_from_dir(args.aa_feature_folder)

    pep_df = pd.read_csv(args.train_file, sep=args.delimiter)

    assert (all([np.issubdtype(pep_df[c], np.number) for c in pep_feature_cols if c != args.pep_col]))

    rep_metrics = []

    for rep in range(1, args.repetitions + 1):
        logging.info('')
        logging.info(f'Training rep {rep} of {args.repetitions} reps...')

        rep_outdir = os.path.join(args.outdir, f'rep{str(rep)}')
        # seed = random.randrange(0, 1000) if seeds is None else int(seeds[rep - 1])
        seed = 1
        logging.info(f"Rep {rep} using seed = {seed}")

        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Start training loop (modified from trainer())
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        metric_rows = []
        for featname in pep_feature_sets_dict:
            logging.info(f"Training feature set {str(featname)} of {str(list(pep_feature_sets_dict.keys()))}")

            feat_set = pep_feature_sets_dict[featname]
            run_out_path = '-'.join([args.run_name, featname]) if args.run_name else featname

            models_dir, configs_dir, eval_dir = make_output_dirs(output_dir=rep_outdir, out_path=run_out_path)

            eval_dict = {"peptide": [], "allele": [], "target": [], "pred_mean": [], "pred_var": [], "fold": []}
            cross_fold_dict = {"fold": [], "ppv": [], "prauc": []}

            # peptide_dataset.set_feat_cols(feat_set)
            # feature_dims = peptide_dataset.feature_dimensions()

            for fold in range(args.folds):
            # for fold in [1,4]:
                logging.info(f"Training fold {str(fold + 1)} of {args.folds}")
                logging.info(f"Decoy mul: {args.decoy_mul}")
                logging.info(f"Decoy ratio: {args.decoy_ratio}")
                logging.info(f"Resampling hits: {args.resampling_hits}")

                train_df = pep_df.loc[(pep_df["set"] == "train") & (pep_df[args.fold_col] == fold)].reset_index(drop=True)
                test_df = pep_df.loc[(pep_df["set"] == "test") & (pep_df[args.fold_col] == fold)].reset_index(drop=True)
                valid_df = pep_df.loc[(pep_df["set"] == "valid") & (pep_df[args.fold_col] == fold)].reset_index(drop=True)

                train_ds = PeptideDatasetTrain(train_df,
                                               pep_col_name=args.pep_col,
                                               allele_col_name=args.allele_col,
                                               target_col_name=args.target_col,
                                               fold_col_name=args.fold_col,
                                               aa_feature_files=aa_feature_files,
                                               hla_encoding_file=args.hla_encoding_file)
                test_ds = PeptideDatasetTrain(test_df,
                                              pep_col_name=args.pep_col,
                                              allele_col_name=args.allele_col,
                                              target_col_name=args.target_col,
                                              fold_col_name=args.fold_col,
                                              aa_feature_files=aa_feature_files,
                                              hla_encoding_file=args.hla_encoding_file)
                valid_ds = PeptideDatasetTrain(valid_df,
                                               pep_col_name=args.pep_col,
                                               allele_col_name=args.allele_col,
                                               target_col_name=args.target_col,
                                               fold_col_name=args.fold_col,
                                               aa_feature_files=aa_feature_files,
                                               hla_encoding_file=args.hla_encoding_file)

                # Sample elements randomly from a given list of ids, no replacement.
                train_subsampler = peptide_nn.PeptideRandomSampler([i for i in range(len(train_ds))], seed)
                test_subsampler = peptide_nn.PeptideRandomSampler([i for i in range(len(test_ds))], seed)
                val_subsampler = peptide_nn.PeptideRandomSampler([i for i in range(len(valid_ds))], seed)

                num_workers = 4#max(1, len(os.sched_getaffinity(0)))
                prefetch_factor = 16 #2**num_workers
                pin_memory = True
                # 16 workers 16 prefetch pin memory True seed 123 2 reps = 1:16x3 min from device = cuda to stat printout
                # 4 workers 16 prefetch pin memory False seed 123 2 reps = 1:18 1:10 1:11
                # 4 workers 16 prefetch pin memory True seed 123 2 reps 1:19 1:06 1:11
                # 16 workers 16 prefetch pin memory False seed 123 2 reps 1:28 1:21 1:25
                # 2 workers 16 prefetch pin memory 2 reps seed 123 = 1:37 1:27 1:40
                # 2 workers 8 prefetch = 1.37 1.33 1.34 
                # 4 workers, 16 prefetch, pin memory True seed 123 = 1.15 1.07 0.59
                
                logging.info(f"Num workers: {num_workers}")
                logging.info(f"Prefetch factor: {prefetch_factor}")
                logging.info(f"Pin memory?: {pin_memory}")
                # Define data loaders for training and testing data in this fold
                trainloader = torch_data.DataLoader(train_ds, 
                                                    batch_size=args.batch_size,
                                                    sampler=train_subsampler,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    prefetch_factor=prefetch_factor,
                                                    persistent_workers=True)
                testloader =  torch_data.DataLoader(test_ds, 
                                                    batch_size=args.batch_size,
                                                    sampler=test_subsampler,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    prefetch_factor=prefetch_factor,
                                                    persistent_workers=True)
                valloader =   torch_data.DataLoader(valid_ds, 
                                                    batch_size=args.batch_size,
                                                    sampler=val_subsampler,
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    prefetch_factor=prefetch_factor,
                                                    persistent_workers=True)
                src_vocab1 = 22
                hla_dim = 4008  # TODO: hard-coding for now
                # src_vocab2 = 15000
                # model = peptide_transformer.OverallModel(src_vocab1, src_vocab2, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
                version=8
                print(f'Running version {version}...')
                model = peptide_transformer.OverallModel_2(src_vocab1, hla_dim, N=6, d_model=22, d_ff=2048, h=2, version=version) # trying diff pos enc style, remove or switch to 2 to change back
                peptide_transformer.initialize_param(model)

                # model = peptide_nn.PeptideNN2(feature_dims, args.dropout_rate)
                if torch.cuda.device_count() > 1:
                    logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
                    model = nn.DataParallel(model)
                model.to(device)
                logging.info(f"Device: {str(device)}")
                # optimizer_dict = peptide_nn.train(model, trainloader, args.learning_rate, args.epochs, device, valloader)
                optimizer_dict = peptide_transformer.train(model, trainloader, args.learning_rate, args.epochs, device, valloader, lr_warmup=4000)

                # Create model and train
                model_config = create_config_dict(device=device, epochs=args.epochs, lr=args.learning_rate,
                                                  dr=args.dropout_rate,
                                                  batch_size=args.batch_size, decoy_mul=args.decoy_mul,
                                                  eval_decoy_ratio=args.decoy_ratio,
                                                  fold=fold, aa_features=args.aa_feature_folder,
                                                  featname=featname, pepfeats_dict=pep_feature_sets_dict, seed=seed)

                peptide_transformer.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)

                _, targets, indices, preds = peptide_transformer.evaluate(model, testloader, args.pred_replicates, device)
                # feature_dims = train_ds.feature_dimensions()
                # assert(feature_dims == test_ds.feature_dimensions() == valid_ds.feature_dimensions())
                #
                # model = peptide_nn.PeptideNN2(feature_dims, args.dropout_rate)
                # if torch.cuda.device_count() > 1:
                #     logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
                #     model = nn.DataParallel(model)
                # model.to(device)
                # logging.info(f"Device: {str(device)}")
                # optimizer_dict = peptide_nn.train(model, trainloader, args.learning_rate, args.epochs, device, valloader, patience=4)
                #
                # # Create model and train
                # model_config = create_config_dict(device=device, epochs=args.epochs, lr=args.learning_rate, dr=args.dropout_rate,
                #                                   batch_size=args.batch_size, decoy_mul=args.decoy_mul,
                #                                   eval_decoy_ratio=args.decoy_ratio,
                #                                   fold=fold, aa_features=aa_feature_files,
                #                                   featname=featname, pepfeats_dict=pep_feature_sets_dict, seed=seed)
                #
                # peptide_nn.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)
                #
                # _, targets, indices, preds = peptide_nn.evaluate(model, testloader, args.pred_replicates, device)
                targets = [t.item() for t in torch.hstack(targets).cpu()]
                indices = [i.item() for i in torch.hstack(indices).cpu()]
                preds = torch.vstack(preds).cpu()

                input_peps, input_alleles = ([test_ds.pep_at(i) for i in indices],
                                             [test_ds.allele_at(i) for i in indices])
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
                f"""Training and evaluation finished for {featname} {os.path.basename(rep_outdir)}.
                             AUC: {auc}
                             PPV: {ppv}
                             PRAUC: {prauc}
                        """)

            metric_rows.append([featname, auc, prauc, ppv])

        eval_metrics = pd.DataFrame(metric_rows, columns=['features', 'auc', 'prauc', 'ppv'])
        eval_metrics.to_csv(os.path.join(rep_outdir, 'eval_metrics.tsv'), sep='\t', index=False)

        rep_metrics.extend(metric_rows)

        logging.info(f"Training finished for rep {rep}. Outputs stored in {rep_outdir}")

    logging.info(f'Writing all_metrics file to {os.path.join(args.outdir, "all_metrics.tsv")}')
    all_metrics = pd.DataFrame(rep_metrics, columns=["features", "auc", "prauc", "ppv"])
    all_metrics.to_csv(os.path.join(args.outdir, 'all_metrics.tsv'), sep='\t', index=False)

    # create summary plots if running multiple reps
    if args.repetitions > 1:
        logging.info('')
        logging.info(f'Creating summary evaluation tables and figures across {args.repetitions} replicates')
        logging.info(f'Writing summary metric table to {os.path.join(args.outdir, "summary_metrics.tsv")}')
        summ_metrics = get_dedup_pep_df(all_metrics)
        summ_metrics.to_csv(os.path.join(args.outdir, 'summary_metrics.tsv'), sep='\t', index=False)

        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_prauc", "std_prauc", args.outdir,
                                                         "prauc_barplot.png")
        TrainingEvaluation.save_feature_comparison_plots(summ_metrics, "mean_ppv", "std_ppv", args.outdir, "ppv_barplot.png")


def trainer(args,
            peptide_dataset: PeptideDatasetTrain,
            val_dataset: PeptideDatasetTrain,
            output_dir: os.PathLike,
            seed: int = 1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    feat_cols = [] if len(args.feat_cols) == 0 else args.feat_cols.split(";")
    featsets_dict = {str(feat_cols): [col for col in feat_cols]} if not args.feat_sets \
        else parse_feature_sets(args.feat_sets)
    # override peptide column input if feature set dict is provided
    pep_feature_cols = list({x for f in list(featsets_dict.values()) if f for x in f}) if args.feat_sets \
        else feat_cols
    pep_feature_cols.append(args.pep_col)

    metric_rows = []

    if args.assign_folds:
        peptide_dataset.reassign_folds(args.folds, seed=seed)

    for featname in featsets_dict:
        logging.info(f"Training feature set {str(featname)} of {str(list(featsets_dict.keys()))}")

        feat_set = featsets_dict[featname]
        run_out_path = '-'.join([args.run_name, featname]) if args.run_name else featname

        models_dir, configs_dir, eval_dir = make_output_dirs(output_dir=output_dir, out_path=run_out_path)

        eval_dict = {"peptide": [], "allele": [], "target": [], "pred_mean": [], "pred_var": [], "fold": []}
        cross_fold_dict = {"fold": [], "ppv": [], "prauc": []}

        peptide_dataset.set_feat_cols(feat_set)
        # feature_dims = peptide_dataset.feature_dimensions()

        if val_dataset is not None:
            val_dataset.set_feat_cols(feat_set)
            # assert (feature_dims == val_dataset.feature_dimensions())

        for fold in range(args.folds):

            logging.info(f"Training fold {str(fold + 1)} of {args.folds}")
            logging.info(f"Decoy mul: {args.decoy_mul}")
            logging.info(f"Decoy ratio: {args.decoy_ratio}")
            logging.info(f"Resampling hits: {args.resampling_hits}")

            train_ids = peptide_dataset.get_train_idxs(fold=fold, decoy_mul=args.decoy_mul, resampling_hits=args.resampling_hits,
                                                       seed=seed)
            test_ids = peptide_dataset.get_test_idxs(fold=fold, decoy_ratio=args.decoy_ratio, seed=seed)

            unique, counts = np.unique(peptide_dataset.pep_df['ha__target'][train_ids], return_counts=True)
            logging.info(f"""Train split:
                            \n{np.asarray((unique, counts))}
                         """)
            # if len(feat_set):
            #     logging.info(peptide_dataset.pep_df.iloc[train_ids].groupby('ha__target')[feat_set].describe())
            # logging.info('')

            unique, counts = np.unique(peptide_dataset.pep_df['ha__target'][test_ids], return_counts=True)
            logging.info(f"""Test split:
                            \n{np.asarray((unique, counts))}
                         """)
            # if len(feat_set):
            #     logging.info(peptide_dataset.pep_df.iloc[test_ids].groupby('ha__target')[feat_set].describe())
            # logging.info('')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = peptide_nn.PeptideRandomSampler(train_ids, seed)
            test_subsampler = peptide_nn.PeptideRandomSampler(test_ids, seed)

            # Define data loaders for training and testing data in this fold
            trainloader = torch_data.DataLoader(peptide_dataset, batch_size=args.batch_size, sampler=train_subsampler)
            testloader = torch_data.DataLoader(peptide_dataset, batch_size=args.batch_size, sampler=test_subsampler)

            try:
                val_subsampler = peptide_nn.PeptideRandomSampler([i for i in range(len(val_dataset))], seed)
                valloader = torch_data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_subsampler)
            except TypeError:
                valloader = None

            src_vocab1 = 22
            hla_dim = 4008 #TODO: hard-coding for now
            # src_vocab2 = 15000
            # model = peptide_transformer.OverallModel(src_vocab1, src_vocab2, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1)
            model = peptide_transformer.OverallModel_2(src_vocab1, hla_dim, N=6, d_model=22, d_ff=2048, h=2)
            peptide_transformer.initialize_param(model)

            # model = peptide_nn.PeptideNN2(feature_dims, args.dropout_rate)
            if torch.cuda.device_count() > 1:
                logging.info(f"Let's use {torch.cuda.device_count()} GPUs!")
                model = nn.DataParallel(model)
            model.to(device)
            logging.info(f"Device: {str(device)}")
            # optimizer_dict = peptide_nn.train(model, trainloader, args.learning_rate, args.epochs, device, valloader)
            optimizer_dict = peptide_transformer.train(model, trainloader, args.learning_rate, args.epochs, device, valloader, lr_warmup=20)

            # Create model and train
            model_config = create_config_dict(device=device, epochs=args.epochs, lr=args.learning_rate, dr=args.dropout_rate,
                                              batch_size=args.batch_size, decoy_mul=args.decoy_mul, eval_decoy_ratio=args.decoy_ratio,
                                              fold=fold, aa_features=args.aa_feature_folder,
                                              featname=featname, pepfeats_dict=featsets_dict, seed=seed)

            peptide_transformer.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)
            # peptide_nn.save_model(model, fold, models_dir, configs_dir, optimizer_dict, model_config)


            _, targets, indices, preds = peptide_transformer.evaluate(model, testloader, args.pred_replicates, device)
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


if __name__ == "__main__":
    # main()
    train_preset_split()
