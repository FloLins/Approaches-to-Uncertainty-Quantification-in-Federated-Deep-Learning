import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def plot_entropy_and_variance(entropy1, entropy1_right, entropy1_wrong, entropy2, variance, variance2, name):
    plt.style.use('ggplot')
    bin = 10
    font = { 'size': 12}

    plt.rc('font', **font)

    fig3, axs3 = plt.subplots(1, 4)
    title3 = 'Entropy of ' + name
    fig3.suptitle(title3)
    axs3[0].set_xlabel("Entropy Value")
    axs3[0].set_ylabel("Relative Frequency")
    axs3[0].set_title('Entropy Normal Data')
    axs3[0].hist(entropy1, weights=np.zeros_like(entropy1) + 1/len(entropy1))
    axs3[1].set_xlabel("Entropy Value")
    axs3[1].set_ylabel("Relative Frequency")
    axs3[1].set_title('Entropy of Right Classified Normal Data')
    axs3[1].hist(entropy1_right, weights=np.zeros_like(entropy1_right) + 1/len(entropy1_right))
    axs3[2].set_xlabel("Entropy Value")
    axs3[2].set_ylabel("Relative Frequency")
    axs3[2].set_title('Entropy of Wrong Classified Normal Data')
    axs3[2].hist(entropy1_wrong, weights=np.zeros_like(entropy1_wrong) + 1/len(entropy1_wrong))
    axs3[3].set_xlabel("Entropy Value")
    axs3[3].set_ylabel("Relative Frequency")
    axs3[3].set_title("Entropy of OOD Data")
    axs3[3].hist(entropy2, weights=np.zeros_like(entropy2) + 1/len(entropy2))

    fig5, axs5 = plt.subplots(2, 2)
    title5 = 'Variance of ' + name
    fig5.suptitle(title5)
    axs5[0, 0].set_xlabel("Variance Value")
    axs5[0, 0].set_ylabel("Frequency")
    axs5[0, 0].set_title('Absolute Variance Normal Data')
    axs5[0, 0].hist(variance, bins=bin)
    axs5[0, 1].set_xlabel("Variance Value")
    axs5[0, 1].set_ylabel("Frequency")
    axs5[0, 1].set_title('Normalized Variance Normal Data')
    axs5[0, 1].hist(variance, weights=np.zeros_like(variance) + 1/len(entropy2))
    axs5[1, 0].set_xlabel("Variance Value")
    axs5[1, 0].set_ylabel("Relative Frequency")
    axs5[1, 0].set_title('Absolute Variance OOD Data')
    axs5[1, 0].hist(variance2, bins=bin)
    axs5[1, 1].set_xlabel("Variance Value")
    axs5[1, 1].set_ylabel("Relative Frequency")
    axs5[1, 1].set_title('Normalized Variance OOD Data')
    axs5[1, 1].hist(variance2, weights=np.zeros_like(variance2) + 1/len(variance2))
    plt.show()


def plot_calibration_curve(curve1, soft1, name):
    logreg_y1, logreg_x1 = calibration_curve(curve1, soft1, n_bins=30)
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--', label="optimal")
    plt.plot(logreg_x1, logreg_y1, marker='.', label="IN-Data")
    fig.suptitle(name, fontsize=20)
    plt.xlabel('Confidence', fontsize=18)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.show()

