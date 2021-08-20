import numpy as np
from utilities import functions
import random
import torch


def get_on_same_size(entropys):#adds mean until all entrys have the same size
    length = []
    for entry in entropys:
        length.append(len(entry))
    max_length = np.max(length)

    for i in range(len(entropys)):
        length = len(entropys[i])
        diff = max_length - length
        for _ in range(diff):
            mean = np.mean(entropys[i], axis=0)
            entropys[i] = np.append(entropys[i], mean)
    return entropys

def sample_on_same_size(matrix1, matrix2): #downsamples bigger matrix to size of smaller matrix
    new_matrix1 = []
    new_matrix2 = []

    for i in range(len(matrix1)):
        if len(matrix1[i]) > len(matrix2[i]):
            bigger = matrix1[i]
            smaller = matrix2[i]
            new_matrix2.append(smaller)
            bigger = random.sample(bigger.tolist(), k=len(smaller))
            new_matrix1.append(bigger)
        else:
            bigger = matrix2[i]
            smaller = matrix1[i]
            new_matrix1.append(smaller)
            bigger = random.sample(bigger.tolist(), k=len(smaller))
            new_matrix2.append(bigger)
    return new_matrix1, new_matrix2


def test_multiple_seeds(model_list, test_set, test_set_ood, pred_number):
    print('[+] TESTING MODEL..')
    all_predictions_in_data, right_predictions_in_data, wrong_predictions_in_data, calibration_in_data = test_multiple_models(model_list, test_set, pred_number)
    print('[+] TESTING OOD MODEL..')
    all_predictions_ood_data, _, _, _ = test_multiple_models(model_list, test_set_ood, pred_number)

    in_data_ood_data_ent_aurocs = compute_multiple_aurocs(all_predictions_in_data['entropies'], all_predictions_ood_data['entropies'])
    in_data_ood_data_var_aurocs = compute_multiple_aurocs(all_predictions_in_data['variances'], all_predictions_ood_data['variances'])

    right_wrong_ent_aurocs = []
    for i in range(3):
        temp1, temp2 = sample_on_same_size(right_predictions_in_data['entropies'], wrong_predictions_in_data['entropies'])
        right_wrong_ent_auroc = compute_multiple_aurocs(temp1, temp2)
        right_wrong_ent_aurocs.append(right_wrong_ent_auroc)

    right_wrong_var_aurocs = []
    for i in range(3):
        temp1, temp2 = sample_on_same_size(right_predictions_in_data['variances'], wrong_predictions_in_data['variances'])
        right_wrong_var_auroc = compute_multiple_aurocs(temp1, temp2)
        right_wrong_var_aurocs.append(right_wrong_var_auroc)
    print(right_wrong_var_aurocs)

    print("#################")
    print(all_predictions_in_data['accuracies'])
    mean_accuracy_in_data = np.mean(all_predictions_in_data['accuracies'], axis=0)
    mean_entropy_in_data =  np.mean(all_predictions_in_data['entropies'])
    mean_variance_in_data = np.mean(all_predictions_in_data['variances'])

    mean_accuracy_ood_data = np.mean(all_predictions_ood_data['accuracies'], axis=0)
    mean_entropy_ood_data =  np.mean(all_predictions_ood_data['entropies'])
    mean_variance_ood_data = np.mean(all_predictions_ood_data['variances'])

    mean_entropy_right_preds = np.mean(get_on_same_size(right_predictions_in_data['entropies']))
    mean_variance_right_preds = np.mean(get_on_same_size(right_predictions_in_data['variances']))
    mean_entropy_wrong_preds = np.mean(get_on_same_size(wrong_predictions_in_data['entropies']))
    mean_variance_wrong_preds = np.mean(get_on_same_size(wrong_predictions_in_data['variances']))

    in_data = {
        "accuracy": mean_accuracy_in_data,
        "accuracy_std": np.std(all_predictions_in_data['accuracies']),
        "entropy": mean_entropy_in_data,
        "entropy_std": np.std(all_predictions_in_data['entropies']),
        "variance": mean_variance_in_data,
        "variance_std":np.std(all_predictions_in_data['variances'])
    }

    ood_data = {
        "accuracy": mean_accuracy_ood_data,
        "accuracy_std": np.std(all_predictions_ood_data['accuracies']),
        "entropy": mean_entropy_ood_data,
        "entropy_std": np.std(all_predictions_ood_data['entropies']),
        "variance": mean_variance_ood_data,
        "variance_std":np.std(all_predictions_ood_data['variances'])
    }

    right_wrong = {
        "entropy_right": mean_entropy_right_preds,
        "entropy_right_std": np.std(right_predictions_in_data['entropies']),
        "variance_right": mean_variance_right_preds,
        "variance_right_std": np.std(right_predictions_in_data['variances']),
        "entropy_wrong": mean_entropy_wrong_preds,
        "entropy_wrong_std": np.std(wrong_predictions_in_data['entropies']),
        "variance_wrong": mean_variance_wrong_preds,
        "variance_wrong_std": np.std(wrong_predictions_in_data['variances'])
    }

    aurocs = {
        "in_data_ood_data_ent": np.mean(in_data_ood_data_ent_aurocs),
        "in_data_ood_data_ent_std": np.std(in_data_ood_data_ent_aurocs),
        "in_data_ood_data_var": np.mean(in_data_ood_data_var_aurocs),
        "in_data_ood_data_var_std": np.std(in_data_ood_data_var_aurocs),
        "right_wrong_ent": np.mean(right_wrong_ent_aurocs),
        "right_wrong_ent_std": np.std(right_wrong_ent_aurocs),
        "right_wrong_var": np.mean(right_wrong_var_aurocs),
        "right_wrong_var_std": np.std(right_wrong_var_aurocs)

    }
    return in_data, ood_data, right_wrong, aurocs, calibration_in_data


def compute_multiple_aurocs(matrix1, matrix2):
    all_aurocs = []
    for i in range(len(matrix1)):
        single_auroc = functions.compute_auroc(matrix1[i], matrix2[i])
        all_aurocs.append(single_auroc)
    return all_aurocs


def test_multiple_models(model_list, test_set, pred_number):
    accuracys, entropys, entropys_right, entropys_wrong, variances, vars_right, vars_wrong = [], [], [], [], [], [], []
    curve, soft = None, None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in model_list:
        model.to(device)
        all_predictions, right_predictions, wrong_predictions, calibration = model.test_model(test_set, pred_number)
        accuracys.append(all_predictions['accuracy'])
        entropys.append(all_predictions['entropy'])
        variances.append(all_predictions['variance'])
        entropys_right.append(right_predictions['entropy'])
        vars_right.append(right_predictions['variance'])
        entropys_wrong.append(wrong_predictions['entropy'])
        vars_wrong.append(wrong_predictions['variance'])

        if curve is None:
            curve = calibration['curve_label']
        else:
            curve = curve + calibration['curve_label']
        if soft is None:
            soft = calibration['softmax_pred']
        else:
            soft = np.concatenate((soft, calibration['softmax_pred']))

    all_predictions = {
        "accuracies": accuracys,
        "entropies": entropys,
        "variances": variances
    }
    right_predictions = {
        "entropies": entropys_right,
        "variances": vars_right
    }
    wrong_predictions = {
        "entropies": entropys_wrong,
        "variances": vars_wrong
    }
    calibration = {
        "curve_label": curve,
        "softmax_pred": soft
    }
    return all_predictions, right_predictions, wrong_predictions, calibration
