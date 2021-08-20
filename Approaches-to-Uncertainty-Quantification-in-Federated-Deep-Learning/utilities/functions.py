import numpy as np
from sklearn.metrics import roc_auc_score


def calculate_variance(tensor):
    '''gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10) '''
    variance = []
    tensor = np.array(tensor)
    #print(tensor.shape)
    for r in range(tensor.shape[1]):
        var = np.var(tensor[:, r, :], axis=0)
        var = np.round(var, decimals=10)
        variance.append(var)
    return np.array(variance)


def calculate_entropy(tensor):
    '''gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10) '''
    entropy = []
    tensor = np.array(tensor)
    final = tensor.mean(0)
    for r in range(final.shape[0]):
        ent_cal = (-final[r, :] * np.log(final[r, :] + 1e-8)).sum()
        entropy.append(ent_cal)
    return np.array(entropy)


def calculate_KLD(tensor):
    '''gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10) '''
    KL = np.zeros((tensor.shape[1]))
    for j in range(tensor.shape[1]):
        for r in range(tensor.shape[0]-1):
            KL[j] += 1/(tensor.shape[0]-1)*compute_kl_divergence(tensor[r,j,:], tensor[r+1,j,:])
    return KL


def compute_kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    p_probs = np.clip(p_probs, 1e-6, 1)
    q_probs = np.clip(q_probs, 1e-6, 1)
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div)


def aleatoric_uncertainty(tensor):
    '''Calculates the uncertainty as (4) in https://openreview.net/pdf?id=Sk_P2Q9sG
    gets a numpy ndarray of shape: number of drawn samples, number of different pictures put in, output nodes (10)
    '''
    aleat = []
    for r in range(tensor.shape[1]):
        aleatoric = np.mean(tensor[:, r, :] * (1 - tensor[:, r, :]), axis=0)
        aleat.append(aleatoric)
    return aleat


def compute_auroc(val1, val2):
    val1 = np.multiply(val1, -1)
    val2 = np.multiply(val2, -1)
    values = np.concatenate((val1, val2))
    score1 = [1] * len(val1)
    score1 = np.array(score1)
    score2 = [0] * len(val2)
    score2 = np.array(score2)
    score = np.concatenate((score1, score2))
    return roc_auc_score(score, values)

