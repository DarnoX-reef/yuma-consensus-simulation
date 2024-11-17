
import torch

weights_epochs_case_1 = []
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
    weights_epochs_case_1.append(W)

weights_epochs_case_2 = []
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
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # All validators support Server 1
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    if epoch >= 1:
        # All validators support Server 2
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
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

weights_epochs_case_6 = []
for epoch in range(80):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # All validators support Server 1
        W[:, 0] = 1.0
    elif epoch == 1:
        # Validator B switches to Server 2
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch >= 2 and epoch <= 40:
        # All validators support Server 2
        W[:, 1] = 1.0
    else:
        # All validators switch back to Server 1
        W[:, 0] = 1.0
    weights_epochs_case_6.append(W)

weights_epochs_case_7 = []
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
    weights_epochs_case_7.append(W)

weights_epochs_case_8 = []
for epoch in range(80):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Validators B and C support Server 1
        W[0, :] = torch.tensor([1.0, 0.001])  # Validator A
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 1:
        # Validators B and C switch to Server 2
        W[0, :] = torch.tensor([1.0, 0.001])  # Validator A
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch >= 2 and epoch <= 40:
        # Validator A copies weights but still supports Server 1 with minimal weight
        W[0, :] = torch.tensor([0.001, 1.0])  # Validator A
        W[1:, 1] = 1.0
    elif epoch == 41:
        # Validators B and C switch back to Server 1
        W[0, :] = torch.tensor([0.001, 1.0])
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Validator A switches to Server 1
        W[0, :] = torch.tensor([1.0, 0.001])  # Validator A
        W[1:, 0] = 1.0
    weights_epochs_case_8.append(W)

cases = [
    {
        'name': 'Case 1',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_1,
        'validators': ['Big vali.', 'Small lazy vali.', 'Small lazier vali.'],
    },
    {
        'name': 'Case 2',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_2,
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
    },
    {
        'name': 'Case 3',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_3,
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
    },
    {
        'name': 'Case 4',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_4,
        'validators': ['Big vali.', 'Small vali.', 'Small vali 2.'],
    },
    {
        'name': 'Case 5',
        'num_epochs': 80,
        'weights_epochs': weights_epochs_case_5,
        'validators': ['Big vali.', 'Small eage-eager vali.', 'Small eager-lazy vali.'],
    },
    {
        'name': 'Case 6',
        'num_epochs': 80,
        'weights_epochs': weights_epochs_case_6,
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
    },
    {
        'name': 'Case 7',
        'num_epochs': 80,
        'weights_epochs': weights_epochs_case_7,
        'validators': ['Big vali.', 'Small eager-eager vali.', 'Small eager-lazy vali.'],
    },
    {
        'name': 'Case 8',
        'num_epochs': 80,
        'weights_epochs': weights_epochs_case_8,
        'validators': ['Big unhonest vali.', 'Small eager-eager vali.', 'Small eager-eager vali 2.'],
    },
]