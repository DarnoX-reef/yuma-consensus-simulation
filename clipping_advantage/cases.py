
import torch

weights_epochs_case_1 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch == 1:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch == 2:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 1] = 1.0  # Validator C -> Server 2
    else:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    weights_epochs_case_1.append(W)

weights_epochs_case_2 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch == 1:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 2:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    weights_epochs_case_2.append(W)

weights_epochs_case_3 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch == 1:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 2:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    else:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    weights_epochs_case_3.append(W)

weights_epochs_case_4 = []
for epoch in range(80):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch == 1:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch == 2:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch >= 3 and epoch <= 40:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    elif epoch == 41:
        W[0, 1] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 2
    else:
        # Subsequent epochs
        W[:, 0] = 1.0 # All validators -> Server 1
    weights_epochs_case_4.append(W)

weights_epochs_case_5 = []
for epoch in range(80):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch == 1:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch == 2:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch >= 3 and epoch <= 40:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    elif epoch == 41:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch == 42:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Subsequent epochs
        W[:, 0] = 1.0 # All validators -> Server 1
    weights_epochs_case_5.append(W)


weights_epochs_cases = {
    'Case 1': weights_epochs_case_1,
    'Case 2': weights_epochs_case_2,
    'Case 3': weights_epochs_case_3,
    'Case 4': weights_epochs_case_4,
    'Case 5': weights_epochs_case_5
}