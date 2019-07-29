#!/usr/env/python
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

max_epoch = 300

def add_plot(ax, vector, label):
    x, y = [], []
    for epoch, value in vector:
        if epoch in x:
            x[epoch - 1] = epoch
            y[epoch - 1] = value
        else:
            x.append(epoch)
            y.append(value)

    ax.plot(x[:max_epoch], y[:max_epoch], label=label)
    ax.legend()


def tolist(vector):
    x, y = [], []
    for epoch, value in vector:
        if epoch in x:
            x[epoch - 1] = epoch
            y[epoch - 1] = value
        else:
            x.append(epoch)
            y.append(value)
    return x, y


def plot_single(fignum, result, dataset, outdir=None):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['loss'], f"single, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_single_{dataset}_loss.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['accuracy'], f"single, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_single_{dataset}_accuracy.png"))

    fn = fignum + 3
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['auc_score'], f"single, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average AUC score")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_single_{dataset}_auc_score.png"))


def plot_iid_dist_full(fignum, title, result, outdir):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['loss'], f"{title}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.savefig(outdir.joinpath(f"{fn}_iid_dist_full_loss.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['accuracy'], f"{title}, chexpert testset")
    add_plot(ax, result['xtest_accuracy'], f"{title}, mimic testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.savefig(outdir.joinpath(f"{fn}_iid_dist_full_accuracy.png"))

    fn = fignum + 3
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['auc_score'], f"{title}, chexpert testset")
    add_plot(ax, result['xtest_auc_score'], f"{title}, mimic testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average AUC score")
    plt.savefig(outdir.joinpath(f"{fn}_iid_dist_full_ave_auc_score.png"))


def plot_noniid_dist_full(fignum, results, prefix, node, dataset, outdir=None):
    result = results[f"{prefix} {node}"]

    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['loss'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_full_{node}_loss.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['accuracy'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_full_{node}_accuracy.png"))

    fn = fignum + 3
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['auc_score'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average auc score")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_full_{node}_ave_auc_score.png"))


def plot_noniid_dist_partial(fignum, results, prefix, node, dataset, outdir=None):
    result = results[f"{prefix} {node}"]

    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['loss'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_partial_{node}_loss.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['accuracy'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_partial_{node}_accuracy.png"))

    fn = fignum + 3
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, result['auc_score'], f"{prefix}, node {node}, {dataset} dataset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average auc score")
    if outdir is not None:
        plt.savefig(outdir.joinpath(f"{fn}_noniid_dist_partial_{node}_ave_auc_score.png"))


def plot_accuracy_comparison(fignum, results, outdir):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, results['chexpert single']['accuracy'], f"single, chexpert trainset, chexpert testset")
    add_plot(ax, results['iid dist full 0']['accuracy'], f"iid, chexpert+mimic trainset, chexpert testset")
    add_plot(ax, results['noniid dist full 0']['accuracy'], f"noniid full, chexpert trainset, chexpert testset")
    add_plot(ax, results['noniid dist partial 0']['accuracy'], f"noniid partial, chexpert trainset, chexpert testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.savefig(outdir.joinpath(f"{fn}_chexpert_accuracy_comparison.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, results['mimic single']['accuracy'], f"single, mimic trainset, mimic testset")
    add_plot(ax, results['iid dist full 0']['xtest_accuracy'], f"iid, chexpert+mimic trainset, mimic testset")
    add_plot(ax, results['noniid dist full 1']['accuracy'], f"noniid full, mimic trainset, mimic testset")
    add_plot(ax, results['noniid dist partial 1']['accuracy'], f"noniid partial, mimic trainset, mimic testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    plt.savefig(outdir.joinpath(f"{fn}_mimic_accuracy_comparison.png"))

def plot_auc_score_comparison(fignum, results, outdir):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, results['chexpert single']['auc_score'], f"single, chexpert trainset, chexpert testset")
    add_plot(ax, results['iid dist full 0']['auc_score'], f"iid, chexpert+mimic trainset, chexpert testset")
    add_plot(ax, results['noniid dist full 0']['auc_score'], f"noniid full, chexpert trainset, chexpert testset")
    add_plot(ax, results['noniid dist partial 0']['auc_score'], f"noniid partial, chexpert trainset, chexpert testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average AUC score")
    plt.savefig(outdir.joinpath(f"{fn}_chexpert_auc_score_comparison.png"))

    fn = fignum + 2
    fig = plt.figure(fn)
    ax = fig.gca()
    add_plot(ax, results['mimic single']['auc_score'], f"single, mimic trainset, mimic testset")
    add_plot(ax, results['iid dist full 0']['xtest_auc_score'], f"iid, chexpert+mimic trainset, mimic testset")
    add_plot(ax, results['noniid dist full 1']['auc_score'], f"noniid full, mimic trainset, mimic testset")
    add_plot(ax, results['noniid dist partial 1']['auc_score'], f"noniid partial, mimic trainset, mimic testset")
    ax.set_xlabel("epoch")
    ax.set_ylabel("average AUC score")
    plt.savefig(outdir.joinpath(f"{fn}_mimic_auc_score_comparison.png"))


def make_dataframe(results):
    exprsets = ('expr set A', 'expr set B', 'expr set C', 'expr set D')
    trainsets = ('chexpert trainset', 'mimic trainset', 'chexpert+mimic trainset')
    testsets = ('chexpert testset', 'mimic testset')

    exprs = {
        ('chexpert single', 'accuracy'): (exprsets[0], trainsets[0], testsets[0]),
        #('chexpert single', 'xtest_accuracy'): (exprsets[0], trainsets[0], testsets[1]),
        ('mimic single', 'accuracy'): (exprsets[0], trainsets[1], testsets[1]),
        #('mimic single', 'xtest_accuracy'): (exprsets[0], trainsets[1], testsets[0]),
        ('iid dist full 0', 'accuracy'): (exprsets[1], trainsets[2], testsets[0]),
        #('iid dist full 0', 'xtest_accuracy'): (exprsets[1], trainsets[2], testsets[1]),
        ('noniid dist full 0', 'accuracy'): (exprsets[2], trainsets[0], testsets[0]),
        #('noniid dist full 0', 'xtest_accuracy'): (exprsets[2], trainsets[0], testsets[1]),
        ('noniid dist full 1', 'accuracy'): (exprsets[2], trainsets[1], testsets[1]),
        #('noniid dist full 1', 'xtest_accuracy'): (exprsets[2], trainsets[1], testsets[0]),
        ('noniid dist partial 0', 'accuracy'): (exprsets[3], trainsets[0], testsets[0]),
        #('noniid dist partial 0', 'xtest_accuracy'): (exprsets[3], trainsets[0], testsets[1]),
        ('noniid dist partial 1', 'accuracy'): (exprsets[3], trainsets[1], testsets[1]),
        #('noniid dist partial 1', 'xtest_accuracy'): (exprsets[3], trainsets[1], testsets[0]),
        ('chexpert single', 'auc_score'): (exprsets[0], trainsets[0], testsets[0]),
        #('chexpert single', 'xtest_auc_score'): (exprsets[0], trainsets[0], testsets[1]),
        ('mimic single', 'auc_score'): (exprsets[0], trainsets[1], testsets[1]),
        #('mimic single', 'xtest_auc_score'): (exprsets[0], trainsets[1], testsets[0]),
        ('iid dist full 0', 'auc_score'): (exprsets[1], trainsets[2], testsets[0]),
        #('iid dist full 0', 'xtest_auc_score'): (exprsets[1], trainsets[2], testsets[1]),
        ('noniid dist full 0', 'auc_score'): (exprsets[2], trainsets[0], testsets[0]),
        #('noniid dist full 0', 'xtest_auc_score'): (exprsets[2], trainsets[0], testsets[1]),
        ('noniid dist full 1', 'auc_score'): (exprsets[2], trainsets[1], testsets[1]),
        #('noniid dist full 1', 'xtest_auc_score'): (exprsets[2], trainsets[1], testsets[0]),
        ('noniid dist partial 0', 'auc_score'): (exprsets[3], trainsets[0], testsets[0]),
        #('noniid dist partial 0', 'xtest_auc_score'): (exprsets[3], trainsets[0], testsets[1]),
        ('noniid dist partial 1', 'auc_score'): (exprsets[3], trainsets[1], testsets[1]),
        #('noniid dist partial 1', 'xtest_auc_score'): (exprsets[3], trainsets[1], testsets[0]),
    }

    accs, aucs = [], []
    for category, data in results.items():
        for metric, measure in data.items():
            keystr = (category, metric)
            if keystr in exprs and (category == 'iid dist full 0' or 'xtest' not in metric):
                labels = exprs[keystr]
            else:
                continue

            x, y = tolist(data[metric])
            max_value = max(y)

            if 'acc' in metric:
                field = {
                    'accuracy': max_value,
                    'expr set': labels[0],
                    'dataset': f"{labels[1]}, {labels[2]}",
                }
                accs.append(pd.DataFrame(field, index=[category]))
            else:
                field = {
                    'average AUC score': max_value,
                    'expr set': labels[0],
                    'dataset': f"{labels[1]}, {labels[2]}",
                }
                aucs.append(pd.DataFrame(field, index=[category]))
    df_acc = pd.concat(accs)
    df_auc = pd.concat(aucs)
    return df_acc, df_auc


def plot_accuracy_barplot(fignum, df, outdir):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    sns.barplot(x="expr set", y="accuracy", hue="dataset", data=df, ax=ax)
    ax.set_ylim(0.8, 0.9)
    plt.savefig(outdir.joinpath(f"{fn}_max_accuracy_comparison.png"))


def plot_auc_score_barplot(fignum, df, outdir):
    fn = fignum + 1
    fig = plt.figure(fn)
    ax = fig.gca()
    sns.barplot(x="expr set", y="average AUC score", hue="dataset", data=df, ax=ax)
    ax.set_ylim(0.7, 0.8)
    plt.savefig(outdir.joinpath(f"{fn}_max_auc_score_comparison.png"))


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    runtime_dirs = {
#        'chexpert trainset': "20190704_noniid_stanford",
#        'mimic trainset': "20190704_noniid_mit",
#        'iid dist full': "20190707_iid_dist",
#        'noniid dist full': "20190707_noniid_dist_full_share",
#        'noniid dist partial': "20190706_noniid_dist_partial_share",

#        'chexpert single': "20190710_20k_single_chexpert",
#        'mimic single': "20190710_20k_single_mimic",
#        'iid dist full': "20190709_20k_iid_dist",
#        'noniid dist full': "20190709_20k_noniid_dist_full_share",
#        'noniid dist partial': "20190710_20k_noniid_dist_partial_share",

#        'chexpert single': "20190718_20k_single_stanford_weight_decay",
#        'mimic single': "20190718_20k_single_mimic_weight_decay",
#        'nih single': "20190718_20k_single_nih_weight_decay",
#        'noniid dist partial': "20190718_20k_noniid_3_datasets_dist_partial_weight_decay",

        'chexpert single': "20190720_20k_single_stanford_dropout",
        'noniid dist full': "20190720_20k_noniid_3_datasets_dist_full_dropout",
        'noniid dist partial': "20190720_20k_noniid_3_datasets_dist_partial_dropout",
    }

    results = {}
    for k, v in runtime_dirs.items():
        assert Path(v).exists(), f"{v} doesn't exist"
        try:
            filepath = Path(v, 'train.pkl').resolve()
            with open(filepath, 'rb') as f:
                results[k] = pickle.load(f)
        except:
            for i in range(3):
                filepath = Path(v, f'train.{i}.pkl').resolve()
                try:
                    with open(filepath, 'rb') as f:
                        results[k + f' {i}'] = pickle.load(f)
                except:
                    continue

    outdir = Path("plots").resolve()
    if not outdir.exists():
        outdir.mkdir(mode=0o755, parents=True, exist_ok=True)

    plot_single(10, results['chexpert single 0'], "chexpert", outdir=None)
    #plot_single(20, results['mimic single 0'], "mimic", outdir=None)
    #plot_single(30, results['nih single 0'], "nih", outdir=None)

    plot_noniid_dist_full(10, results, "noniid dist full", 0, "chexpert", outdir=None)
    plot_noniid_dist_partial(10, results, "noniid dist partial", 0, "chexpert", outdir)
    #plot_noniid_dist_partial(20, results, "noniid dist partial", 1, "mimic", outdir)
    #plot_noniid_dist_partial(30, results, "noniid dist partial", 2, "nih", outdir)

    """
    plot_iid_dist_full(30, 'chexpert+mimic trainset', results['iid dist full 0'], output_dir)

    plot_noniid_dist_full0(40, 'chexpert trainset', results['noniid dist full 0'], output_dir)
    plot_noniid_dist_full1(40, 'mimic trainset', results['noniid dist full 1'], output_dir)

    plot_noniid_dist_partial0(50, 'chexpert trainset', results['noniid dist partial 0'], output_dir)
    plot_noniid_dist_partial1(50, 'mimic trainset', results['noniid dist partial 1'], output_dir)

    plot_accuracy_comparison(60, results, output_dir)
    plot_auc_score_comparison(70, results, output_dir)

    df_acc, df_auc = make_dataframe(results)
    print(df_acc)
    print(df_auc)

    plot_accuracy_barplot(100, df_acc, output_dir)
    plot_auc_score_barplot(110, df_auc, output_dir)
    """

