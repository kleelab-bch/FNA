'''
Junbong Jang
10/3/2020

Helper functions to visualize bootstrapping results
'''

import numpy as np
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score
import math

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

font = {'size':'11', 'weight':'normal',}
matplotlib.rc('font', **font)

def distribution_mean_and_error(box_counts_series, sample_size):
    mean = box_counts_series.mean()
    std = box_counts_series.std()
    standard_err = std / math.sqrt(sample_size)

    mean = round(mean,3)
    std = round(std,3)

    print('mean', mean)
    print('sample standard deviation', std)
    print('standard error', standard_err)
    print(len(box_counts_series.loc[(box_counts_series >= mean-std) & (box_counts_series <= mean+std)]))
    print(len(box_counts_series.loc[(box_counts_series >= mean-standard_err) & (box_counts_series <= mean+standard_err)]))
    print()
    return mean, std


def plot_histogram(box_counts_df, test_image_names, save_base_path):
    '''

    :param box_counts_df:
    :return:
    '''
    gt_mean, gt_std = distribution_mean_and_error(box_counts_df[0], sample_size=len(test_image_names))
    pred_mean, pred_std = distribution_mean_and_error(box_counts_df[1], sample_size=len(test_image_names))

    bins = np.histogram(np.hstack((box_counts_df[0], box_counts_df[1])), bins=40)[1]  # get the bin edges
    fig, ax = plt.subplots()
    plt.hist(box_counts_df[0], bins=bins, rwidth=0.9, alpha=0.5, label='Ground Truth (GT)')
    plt.hist(box_counts_df[1], bins=bins, rwidth=0.9, alpha=0.5, label='Prediction (Pred)')

    ax.text(0.68, 0.74, f'GT Mean={gt_mean}\nGT SD={gt_std}\nPred Mean={pred_mean}\nPred SD={pred_std}', color='black',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)

    # plt.title('Distribution of Samples of Follicular Clusters', fontsize='x-large')
    plt.xlabel('Number of Follicular Clusters', fontsize='large')
    plt.ylabel(f'Frequency', fontsize='large')
    plt.legend(loc='upper right')

    # ---- for two separate histograms
    # https://stackoverflow.com/questions/23617129/matplotlib-how-to-make-two-histograms-have-the-same-bin-width

    # hist = box_counts_df.hist(bins=10, rwidth=0.9)

    # fig = plt.gcf()
    # fig.ylim((0, 250))
    # fig.suptitle('Distribution of Samples of Follicular Clusters', fontsize='x-large')
    #
    # hist[0,0].set_xlabel('Number of Follicular Clusters', fontsize='large')
    # hist[0,0].set_ylabel(f'Frequency', fontsize='large')
    # hist[0, 0].set_title('Ground Truth')
    # hist[0,0].grid(False)
    #
    # hist[0,1].set_xlabel('Number of Follicular Clusters', fontsize='large')
    # hist[0,1].set_ylabel(f'Frequency', fontsize='large')
    # hist[0,1].set_title('Prediction')
    # hist[0,1].grid(False)
    #
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.88)

    plt.xlim( right=200)
    figure_postprocess(False)
    plt.savefig(save_base_path + 'data_histogram.svg')
    plt.close()


def plot_scatter(box_counts_df, ground_truth_min_follicular, save_base_path):
    """
    box_counts_df[0] is ground truth, box_counts_df[1] is prediction
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.scatter(box_counts_df[0], box_counts_df[1], alpha=0.1, s=3)
    plt.axvline(x=ground_truth_min_follicular, c='r')
    plt.axhline(y=ground_truth_min_follicular, c='r')

    # linear regression of scatter plot
    reg = LinearRegression().fit(box_counts_df[0].to_numpy().reshape(-1, 1), box_counts_df[1].to_numpy())
    r_squared = reg.score(box_counts_df[0].to_numpy().reshape(-1, 1), box_counts_df[1].to_numpy())
    m = reg.coef_[0]
    b = reg.intercept_

    plt.plot(box_counts_df[0], m * box_counts_df[0] + b)
    ax.text(0.17, 0.9, f'y=x*{round(m,3)}+{round(b,3)}\n$R^2$={round(r_squared,3)}', color='#1f77b4',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)


    # plt.title('Ground Truth Vs. Predicted', fontsize='x-large')
    plt.xlabel('Ground Truth Number of Follicular Clusters', fontsize='large')
    plt.ylabel('Predicted Number of Follicular Clusters', fontsize='large')

    plt.xlim(left=0, right=150) # right=1000
    plt.ylim(bottom=0, top=150)  # top=1000
    figure_postprocess()
    plt.savefig(save_base_path + f'data_scatter_threshold_{ground_truth_min_follicular}.svg')
    plt.close()


def plot_roc_curve(y_true, y_pred, save_base_path):
    '''
    Refered to
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python
    :param y_true:
    :param y_pred:
    :return:
    '''
    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_true))]

    # calculate scores
    ns_auc = roc_auc_score(y_true, ns_probs)
    lr_auc = roc_auc_score(y_true, y_pred)

    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('ROC AUC=%.3f' % (lr_auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_pred)

    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=f'Follicular Cluster Detection\nROC AUC=%.3f' % (lr_auc))

    # plt.title('Slide Pass/Fail ROC curve')
    plt.xlabel('False Positive Rate', fontsize='large')
    plt.ylabel('True Positive Rate', fontsize='large')

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    figure_postprocess()
    plt.savefig(save_base_path + 'roc_curve.svg')
    plt.close()


def plot_precision_recall_curve(y_true, y_pred, save_base_path):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    lr_f1, lr_auc = f1_score(y_true, y_pred), auc(lr_recall, lr_precision)

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.',
             label=f'Follicular Cluster Detection\nf1={round(lr_f1,3)} auc={round(lr_auc,3)}')

    # plt.title('Slide Pass/Fail Precision-Recall curve')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0)
    plt.ylim(bottom=no_skill)
    plt.legend()
    figure_postprocess()
    plt.savefig(save_base_path + 'precision_recall_curve.svg')
    plt.close()


def plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, ground_truth_min_follicular, save_base_path):
    no_skill, no_skill_auc = helper_plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, 'Follicular Cluster Detection')

    # plt.title('Slide Pass/Fail Precision-Recall curve', fontsize='x-large')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0, right=1.02)
    plt.ylim(bottom= math.floor(no_skill*100)/100)
    plt.legend()
    figure_postprocess()
    plt.savefig(save_base_path + f'precision_recall_curve_at_thresholds_threshold_{ground_truth_min_follicular}.svg')
    plt.close()


def plot_comparison_precision_recall_curve_at_thresholds(y_true, precision_list1, recall_list1,
                                                         precision_list2, recall_list2,
                                                         precision_list3, recall_list3, ground_truth_min_follicular, save_base_path, save_prefix=''):
    no_skill, no_skill_auc = helper_plot_precision_recall_curve_at_thresholds(y_true, precision_list1, recall_list1, 'MTL only')
    no_skill2, _ = helper_plot_precision_recall_curve_at_thresholds(y_true, precision_list2, recall_list2, 'Faster R-CNN only')
    no_skill3, _ = helper_plot_precision_recall_curve_at_thresholds(y_true, precision_list3, recall_list3, 'FNA-Net')

    assert no_skill == no_skill2
    assert no_skill == no_skill3

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill AUC={round(no_skill_auc, 3)}', lw=2)

    # plt.title('Slide Pass/Fail Precision-Recall curve', fontsize='x-large')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0, right=1.02)
    # plt.ylim(bottom= math.floor(no_skill*100)/100)
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0.05))
    figure_postprocess()
    plt.savefig(save_base_path + f'{save_prefix}_precision_recall_curve_at_thresholds_threshold_{ground_truth_min_follicular}.svg')
    plt.close()


def helper_plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, label_string):
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    # include both endpoints
    precision_list = precision_list + [1,no_skill]
    recall_list = recall_list + [0,1]

    # sort them
    recall_sort_index = np.argsort(recall_list)
    precision_list = [precision_list[i] for i in recall_sort_index]
    recall_list = [recall_list[i] for i in recall_sort_index]

    no_skill_auc = auc([0, 1], [no_skill, no_skill])
    lr_auc = auc(recall_list, precision_list)

    plt.plot(recall_list, precision_list, marker='.',
             label=f'{label_string}\nAUC={round(lr_auc, 3)}', lw=2)

    return no_skill, no_skill_auc


def plot_performance_at_thresholds(predicted_min_follicular_list, precision_list, recall_list, f1_list, ground_truth_min_follicular, save_base_path, save_prefix=''):
    assert len(predicted_min_follicular_list) == len(precision_list)
    assert len(predicted_min_follicular_list) == len(recall_list)
    assert len(predicted_min_follicular_list) == len(f1_list)

    plt.plot(predicted_min_follicular_list, precision_list, marker='.', label='Precision', lw=2)
    plt.plot(predicted_min_follicular_list, recall_list, marker='.', label='Recall', lw=2)
    plt.plot(predicted_min_follicular_list, f1_list, marker='.', label='F1', lw=2)

    # plt.title('Performance at different Thresholds', fontsize='x-large')
    plt.xlabel('Minimum Predicted Follicular Clusters to Pass', fontsize='large')
    plt.ylabel('Performance', fontsize='large')

    # plt.ylim(bottom=0.8, top=1)
    # plt.xlim(left=0, right=len(predicted_min_follicular_list))
    plt.legend()
    figure_postprocess()
    plt.savefig(save_base_path + f'{save_prefix}_performance_at_thresholds_threshold_{ground_truth_min_follicular}.svg')
    plt.close()

def figure_postprocess(use_grid=True):
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(use_grid)