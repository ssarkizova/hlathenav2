import os
import random

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (PrecisionRecallDisplay, auc, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)


def make_subdir(output_dir, subdir):
    subdir_path = os.path.join(output_dir,subdir)
    os.makedirs(subdir_path)
    return subdir_path


class TrainingEvaluation:
    def __init__(self, eval_df, outdir=None, decoy_ratio=True, seed=1):

        if decoy_ratio:
            self.hit_eval_df = eval_df[eval_df['target']==1]
            self.decoy_eval_df = \
                eval_df[eval_df['target']==0].sample(
                    1000*len(self.hit_eval_df), random_state=seed, replace=True
                )
            self.eval_df = pd.concat([self.hit_eval_df,self.decoy_eval_df])
        else:
            self.hit_eval_df = eval_df[eval_df['target']==1]
            self.decoy_eval_df = eval_df[eval_df['target']==0]
            self.eval_df = eval_df

        self.y_test = self.eval_df['target']
        self.y_pred = self.eval_df['pred_mean']
        self._check_format()
        self.confusion_matrix = \
            self.confusion_to_df(confusion_matrix(self.y_test, [round(p) for p in self.y_pred]))

        self.outdir = "" if outdir is None else outdir




    def get_ppv(self):
        n = len(self.hit_eval_df)
        sorted_df = self.eval_df.sort_values('pred_mean', ascending=False).head(n)
        return len(sorted_df[sorted_df['target']==1])/n


    def confusion_to_df(self, conmat):
        val = np.mat(conmat)
        classnames = [0,1]
        df_cm = pd.DataFrame(
                val, index=classnames, columns=classnames,
            )
        return df_cm

    def create_confusion_matrix_heatmap(self):
        fig, heatmap = plt.subplots(1)
        heatmap = sns.heatmap(self.confusion_matrix, annot=True, cmap="Blues")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        if self.outdir:
            fig = heatmap.get_figure()
            fig.savefig(os.path.join(self.outdir, 'confusion_matrix.png'))

    def get_auc(self):
        auc = roc_auc_score(self.y_test,self.y_pred)
        return auc

    def get_prauc(self):
        # Data to plot precision - recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred)
        # Use AUC function to calculate the area under the curve of precision recall curve
        auc_precision_recall = auc(recall, precision)
        return auc_precision_recall


    def plot_precision_recall(self, title=""):
        display = \
            PrecisionRecallDisplay.from_predictions(self.y_test,
                                                    self.y_pred,
                                                    pos_label=1,
                                                    name=title)
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        prauc_text = 'PRAUC: {prauc}'.format(prauc=self.get_prauc())
        display.ax_.annotate(prauc_text, xy=(2, 1), xytext=(3, 1.5))
        display.plot()
        plt.show()
        if self.outdir:
            display.figure_.savefig(os.path.join(self.outdir, 'precision_recall_curve.png'))


    def plot_roc_curve(self):
        """
        plots the roc curve based of the probabilities
        """
        fig, ax = plt.subplots(1)
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        if self.outdir:
            fig = ax.get_figure()
            fig.savefig(os.path.join(self.outdir, 'roc_curve.png'))

    def save_feature_comparison_plots(metric_df, meancol, stdcol, outdir, outfile):
        fig1, ax1 = plt.subplots()
        mean = metric_df[meancol]
        std = metric_df[stdcol]

        mean.plot(kind='bar',
                  yerr=std,
                  colormap='OrRd_r',
                  edgecolor='black',
                  grid=False,
                  figsize=(5,5),
                  ax=ax1,
                  position=0.45,
                  error_kw=dict(ecolor='black',
                                elinewidth=0.5),
                  width=0.8)
        fig1.savefig(os.path.join(outdir, outfile))

    def create_score_distribution_plot(self):
        ## Score distribution from equally sampled binders and decoys
        ax = self.y_pred.hist()
        if self.outdir:
            fig = ax.get_figure()
            fig.savefig(os.path.join(self.outdir, 'score_distribution.png'))

    def binomial_var(self, x, A):
        return x*(1-x) * A

    def create_variance_plot(self):
        ## Score distribution from equally sampled binders and decoys
        y_values, bin_edges, _ = plt.hist(self.y_pred, density=True, bins=50)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.plot(self.y_pred, self.eval_df['pred_var'], '.')
        popt, pcov = sp.optimize.curve_fit(self.binomial_var, self.y_pred, self.eval_df['pred_var'])
        ax.plot(bin_edges, self.binomial_var(bin_edges, popt[0]))
        if self.outdir:
            fig.savefig(os.path.join(self.outdir,'variance_plot.png'))

    def _check_format(self):
        assert(set(self.eval_df.columns).issuperset(["peptide", "target", "pred_mean", "pred_var", "fold"]))
