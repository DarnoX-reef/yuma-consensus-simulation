
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
for epoch in range(40):
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
    elif epoch >= 3 and epoch <= 20:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    elif epoch == 21:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch == 22:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Subsequent epochs
        W[:, 0] = 1.0 # All validators -> Server 1
    weights_epochs_case_5.append(W)

weights_epochs_case_6 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # All validators support Server 1
        W[:, 0] = 1.0
    elif epoch == 1:
        # Validator B switches to Server 2
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 2:
        W[0, 1] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch >= 3 and epoch <= 20:
        # All validators support Server 2
        W[:, 1] = 1.0
    else:
        # All validators switch back to Server 1
        W[:, 0] = 1.0
    weights_epochs_case_6.append(W)

weights_epochs_case_7 = []
for epoch in range(40):
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
    elif epoch >= 3 and epoch <= 20:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    elif epoch == 21:
        W[0, 1] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 2
    else:
        # Subsequent epochs
        W[:, 0] = 1.0 # All validators -> Server 1
    weights_epochs_case_7.append(W)

weights_epochs_case_8 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Validators B and C support Server 1
        W[0, :] = torch.tensor([0.999, 0.001])  # Validator A
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 1:
        # Validators B and C switch to Server 2
        W[0, :] = torch.tensor([0.999, 0.001])  # Validator A
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 1] = 1.0  # Validator C -> Server 2
    elif epoch >= 2 and epoch <= 20:
        # Validator A copies weights but still supports Server 1 with minimal weight
        W[0, :] = torch.tensor([0.001, 0.999])  # Validator A
        W[1:, 1] = 1.0
    elif epoch == 21:
        # Validators B and C switch back to Server 1
        W[0, :] = torch.tensor([0.001, 0.999])
        W[1, 0] = 1.0  # Validator B -> Server 1
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Validator A switches to Server 1
        W[0, :] = torch.tensor([0.999, 0.001])  # Validator A
        W[1:, 0] = 1.0
    weights_epochs_case_8.append(W)

weights_epochs_case_9 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    W[:, 1] = 1.0 # All validators -> Server 2
    weights_epochs_case_9.append(W)

weights_epochs_case_10 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # Initialy consensus is achieved by all Validators
        W[:, 0] = 1.0
    elif epoch >=1 and epoch < 10:
        W[0, 0] = 1.0  # Validator A -> Server 1
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    elif epoch == 10:
        W[0, 1] = 1.0  # Validator A -> Server 2
        W[1, 1] = 1.0  # Validator B -> Server 2
        W[2, 0] = 1.0  # Validator C -> Server 1
    else:
        # Subsequent epochs
        W[:, 1] = 1.0  # All validators -> Server 2
    weights_epochs_case_10.append(W)

weights_epochs_case_11 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch < 20:
        # Server 1
        W[0, 0] = 0.3
        W[1, 0] = 0.6
        W[2, 0] = 0.61
        # Server 2
        W[0, 1] = 0.7
        W[1, 1] = 0.4
        W[2, 1] = 0.39
    else:
        # Server 1
        W[0, 0] = 0.3 
        W[1, 0] = 0.6
        W[2, 0] = 0.3 
        # Server 2
        W[0, 1] = 0.7
        W[1, 1] = 0.4
        W[2, 1] = 0.61
    weights_epochs_case_11.append(W)

weights_epochs_case_12 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch == 0:
        # All Validators support server 1
        W[0, 0] = 1.0
        W[1, :] = torch.tensor([0.999, 0.001])
        W[2, 0] = 1.0
    elif epoch >= 1 and epoch <= 20:
        # All Validators support server 2
        W[0, 1] = 1.0
        W[1, :] = torch.tensor([0.001, 0.999])
        W[2, 1] = 1.0
    else:
        # All Validators support server 1
        W[0, 0] = 1.0
        W[1, :] = torch.tensor([0.999, 0.001])
        W[2, 0] = 1.0
    weights_epochs_case_12.append(W)

weights_epochs_case_13 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch <= 20:
        W[0, 1] = 1.0
        W[1, :] = torch.tensor([0.5, 0.5])
        W[2, 1] = 1.0
    else:
        W[0, 1] = 1.0
        W[1, :] = torch.tensor([0.5, 0.5])
        W[2, :] = torch.tensor([0.5, 0.5])
    weights_epochs_case_13.append(W)

weights_epochs_case_14 = []
for epoch in range(40):
    W = torch.zeros(3, 2)
    if epoch <= 20:
        W[:, 0] = 1.0 # All validators -> Server 1
    else:
        W[0, 1] = 1.0 # Validator A -> Server 2
        W[1, 1] = 1.0 # Validator B -> Server 2
    weights_epochs_case_14.append(W)


stakes_epochs_case_1 = torch.tensor([0.8, 0.1, 0.1])

stakes_epochs_case_2 = []
for epoch in range(40):
    if epoch >= 0 and epoch <= 5:
        stakes = torch.tensor([0.8, 0.1, 0.1])
    else:
        stakes = torch.tensor([0.8, 0.2, 0.0]) # Validator C joins to Validator B
    stakes_epochs_case_2.append(stakes)

stakes_epochs_case_3 = torch.tensor([0.49, 0.49, 0.02])

analysis_dict = {
    'Case 1': {
        'weights': "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        'dividends': "Vestibulum at sem vitae sapien convallis aliquet non ut metus.",
        'bonds': "Proin pharetra nunc vel orci cursus, et ultrices nulla faucibus.",
    },
    'Case 2': {
        'weights': "Nullam hendrerit nisl in orci cursus fermentum eu in quam.",
        'dividends': "Aliquam erat volutpat, sed tempus massa sed aliquet varius.",
        'bonds': "Fusce vehicula urna at libero luctus, eget pretium arcu pulvinar.",
    },
    'Case 3': {
        'weights': "Sed ut perspiciatis unde omnis iste natus error sit voluptatem.",
        'dividends': "Quisque sodales justo id sapien sodales tincidunt eu in magna.",
        'bonds': "Aenean faucibus nisi nec sem vehicula, vitae blandit lorem euismod.",
    },
    'Case 4': {
        'weights': "Vivamus pellentesque neque at risus aliquet, sit amet tempus libero.",
        'dividends': "Ut id lorem vel nisi gravida bibendum eget sit amet eros.",
        'bonds': "Integer placerat sapien vel eros consectetur, sed dignissim risus feugiat.",
    },
    'Case 5': {
        'weights': "Donec accumsan lacus id lectus bibendum, at pulvinar nisi tempor.",
        'dividends': "Curabitur nec urna nec sapien rhoncus fermentum sed in odio.",
        'bonds': "Maecenas sit amet urna quis nisi laoreet tincidunt a ut dui.",
    },
    'Case 6': {
        'weights': "Suspendisse potenti, ut malesuada nisl sit amet pulvinar feugiat.",
        'dividends': "Praesent ut velit ut lorem vehicula convallis vitae ac sapien.",
        'bonds': "Nunc nec lacus vel urna facilisis ultricies ut sit amet leo.",
    },
    'Case 7': {
        'weights': "Phasellus vitae lectus eget dolor euismod accumsan vitae id nulla.",
        'dividends': "Ut faucibus urna non nisi laoreet, vel feugiat augue tincidunt.",
        'bonds': "Morbi scelerisque enim vel neque tincidunt, sit amet lacinia ipsum gravida.",
    },
    'Case 8': {
        'weights': "Curabitur at sapien in orci tincidunt fermentum at et est.",
        'dividends': "Mauris hendrerit nisi ut augue placerat, ac varius mi laoreet.",
        'bonds': "Etiam eleifend libero id purus tincidunt, in malesuada eros molestie.",
    },
    'Case 9': {
        'weights': "Fusce fringilla orci vel erat scelerisque, eget varius enim dictum.",
        'dividends': "Pellentesque ultricies nisl vel ligula dapibus malesuada vitae a elit.",
        'bonds': "Sed laoreet erat eget erat pellentesque, ac fermentum ligula tincidunt.",
    },
    'Case 10': {
        'weights': "Sed nec velit nec mi ultricies tincidunt eu nec libero.",
        'dividends': "Sed nec velit nec mi ultricies",
        'bonds': "Sed nec velit nec mi ultricies",
    },
    'case_11': {
        'weights': "Sed nec velit nec mi ultricies tincidunt eu nec libero.",
        'dividends': "Sed nec velit nec mi ultricies",
        'bonds': "Sed nec velit nec mi ultricies",
    },
    'Case 12': {
        'weights': "Sed nec velit nec mi",
        'dividends': "Sed nec velit nec mi",
        'bonds': "Sed nec velit nec mi",
    },
    'Case 13': {
        'weights': "Sed nec velit nec mi",
        'dividends': "Sed nec velit nec mi",
        'bonds': "Sed nec velit nec mi",
    },
    'Case 14': {
        'weights': "Sed nec velit nec mi",
        'dividends': "Sed nec velit nec mi",
        'bonds': "Sed nec velit nec mi",
    },
}


cases = [
    {
        'name': 'Case 1 - kappa moves first',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_1,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 1'],
        'validators': ['Big vali.', 'Small lazy vali.', 'Small lazier vali.'],
        'reset_bonds': False,
    },
    {
        'name': 'Case 2 - kappa moves second',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_2,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 2'],
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
        'reset_bonds': False,
    },
    {
        'name': 'Case 3 - kappa moves third',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_3,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 3'],
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
        'reset_bonds': False,
    },
    {
        'name': 'Case 4 - all validators switch',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_4,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 4'],
        'validators': ['Big vali.', 'Small vali.', 'Small vali 2.'],
        'reset_bonds': False,
        
    },
    {
        'name': 'Case 5 - kappa moves second, then third',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_5,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 5'],
        'validators': ['Big vali.', 'Small eager-eager vali.', 'Small eager-lazy vali.'],
        'reset_bonds': True,
        'reset_bonds_index': 1,
        'reset_bonds_epoch': 20
    },
    {
        'name': 'Case 6 - kappa moves second, then all validators switch',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_6,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 6'],
        'validators': ['Big vali.', 'Small eager vali.', 'Small lazy vali.'],
        'reset_bonds': True,
        'reset_bonds_index': 0,
        'reset_bonds_epoch': 21,
    },
    {
        'name': 'Case 7 - big vali moves late, then all but one small vali moves late',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_7,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 7'],
        'validators': ['Big vali.', 'Small eager-lazy vali.', 'Small eager-eager vali.'],
        'reset_bonds': True,
        'reset_bonds_index': 0,
        'reset_bonds_epoch': 21,
    },
    {
        'name': 'Case 8 - big vali moves late, then late',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_8,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 8'],
        'validators': ['Big dishonest lazy vali.', 'Small eager-eager vali.', 'Small eager-eager vali 2.'],
        'reset_bonds': True,
        'reset_bonds_index': 1,
        'reset_bonds_epoch': 20
    },
    {
        'name': 'Case 9 - small validators merged in e5',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_9,
        'stakes_epochs': stakes_epochs_case_2,
        'analysis': analysis_dict['Case 9'],
        'validators': ['Big vali.', 'Small vali.', 'Small vali 2.'],
        'reset_bonds': False,
    },
    {
        'name': 'Case 10 - kappa delayed',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_10,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 10'],
        'validators': ['Big delayed vali.', 'Small eager vali.', 'Small lazy vali.'],
        'reset_bonds': False,
    },
    {
        'name': 'Case 11 - clipping bug',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_11,
        'stakes_epochs': [stakes_epochs_case_3] * 40,
        'analysis': analysis_dict['case_11'],
        'validators': ['Big vali. 1', 'Big vali. 2', 'Small vali.'],
        'reset_bonds': True,
        'reset_bonds_index': 1,
        'reset_bonds_epoch': 20
    },
    {
        'name': 'Case 12 - all validators switch, but small validators support alt miner with minimal weight',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_12,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 12'],
        'validators': ['Big vali.', 'Small dishonest vali.', 'Small vali.'],
        'reset_bonds': True,
        'reset_bonds_index': 1,
        'reset_bonds_epoch': 20
    },
    {
        'name': 'Case 13 - Big vali supports server 2, small validator/s support server 1',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_13,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 13'],
        'validators': ['Big vali.', 'Small vali.', 'Small vali 2.'],
        'reset_bonds': True,
        'reset_bonds_index': 0,
        'reset_bonds_epoch': 20
    },
    {   
        'name': 'Case 14 - Small lazy validator supports server 1 while others switch to server 2',
        'num_epochs': 40,
        'weights_epochs': weights_epochs_case_14,
        'stakes_epochs': [stakes_epochs_case_1] * 40,
        'analysis': analysis_dict['Case 14'],
        'validators': ['Big vali.', 'Small vali.', 'Small lazy vali.'],
        'reset_bonds': False,
    }
]