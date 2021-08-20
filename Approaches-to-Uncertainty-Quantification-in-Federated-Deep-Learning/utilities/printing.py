def print_data(data):
    print("Accuracy: " + str(data["accuracy"]))
    print("Accuracy-STD: " + str(data["accuracy_std"]))
    print("Entropy: " + str(data["entropy"]))
    print("Entropy-STD: " + str(data["entropy_std"]))
    print("Variance: " + str(data["variance"]))
    print("Variance-STD: " + str(data["variance_std"]))


def print_right_wrong(data):
    print("Entropy-Right: " + str(data["entropy_right"]))
    print("Entropy-Right-STD: " + str(data["entropy_right_std"]))
    print("Entropy-Wrong: " + str(data["entropy_wrong"]))
    print("Entropy-Wrong-STD: " + str(data["entropy_wrong_std"]))
    print("Variance-Right: " + str(data["variance_right"]))
    print("Variance-Right-STD: " + str(data["variance_right_std"]))
    print("Variance-Wrong: " + str(data["variance_wrong"]))
    print("Variance-Wrong-STD: " + str(data["variance_wrong_std"]))


def print_aurocs(data):
    print("IN-Data-OOD-Data-Ent: " + str(data["in_data_ood_data_ent"]))
    print("IN-Data-OOD-Data-Ent-STD: " + str(data["in_data_ood_data_ent_std"]))
    print("IN-Data-OOD-Data-Var: " + str(data["in_data_ood_data_var"]))
    print("IN-Data-OOD-Data-Var-STD: " + str(data["in_data_ood_data_var_std"]))
    print("Right-Wrong-Ent: " + str(data["right_wrong_ent"]))
    print("Right-Wrong-Ent-STD: " + str(data["right_wrong_ent_std"]))
    print("Right-Wrong-Var: " + str(data["right_wrong_var"]))
    print("Right-Wrong-var-STD: " + str(data["right_wrong_var_std"]))
