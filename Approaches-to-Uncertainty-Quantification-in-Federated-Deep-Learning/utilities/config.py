MODEL_PARAMETERS = {'model_type': 'NN',
                    'epochs_per_episode': 10,
                    'prediction_number': 10,
                    'batch_size': 100,
                    'number_of_communications': 10,
                    'amount_of_worker': 20,
                    'amount_of_communicators': 4,
                    'dataset': "MNIST",
                    'ood_dataset': "KMNIST",
                    'lr': None, #Value set by ech experiment
                    'lam': 1e-5,
                    'dropout_rate': 0.5,
                    'dropout_rate_increase': 0.1,
                    'repetitions_for_ensemble': 4
                    }

#Do not change these! They are for controlling different Compatiations like swag or MC Dropout
CONTROLL_PARAMETERS = {'Train': True,
                       'plotting': False,
                       'dropOut': False,
                       'diffrerent_dropout_rates': False,
                       'swag': False,
                       'optimisation:': False
                       }


def load_config():
    return MODEL_PARAMETERS, CONTROLL_PARAMETERS

