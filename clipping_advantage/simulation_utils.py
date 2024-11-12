import torch
import matplotlib.pyplot as plt
import numpy as np  
import math
import itertools
import pandas as pd
from IPython.display import display, Markdown
from cases import weights_epochs_cases
import logging


def Yuma(W, S, B_old=None, kappa=0.5, bond_penalty=1, bond_alpha=0.1, liquid_alpha = False, alpha_high = 0.9, alpha_low = 0.7, precision = 100000, override_consensus_high = None, override_consensus_low = None):
    # === Weight === 
    W = (W.T / (W.sum(dim=1) +  1e-6)).T
    
    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1
        c_low = 0
        
        while (c_high - c_low) > 1/precision:
            c_mid = (c_high + c_low) / 2

            _c_sum = (miner_weight > c_mid) * S
            if sum(_c_sum) > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65535).int() / 65535

    # === Consensus cliped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = (W_clipped).sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - bond_penalty) * W + bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = None
    if liquid_alpha:
        if override_consensus_high == None:
            consensus_high = C.quantile(0.75)
        else:
            consensus_high = override_consensus_high 
        
        if override_consensus_low == None:
            consensus_low = C.quantile(0.25)
        else:
            consensus_low = override_consensus_low 

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)
        a = (math.log(1/alpha_high - 1) - math.log(1/ alpha_low - 1) ) / (consensus_low - consensus_high) 
        b = math.log(1/ alpha_low - 1) + a * consensus_low 
        alpha = 1 / (1 + math.e **(-a *C + b)) # alpha to the old weight
        bond_alpha = 1 - torch.clip(alpha, alpha_low, alpha_high)

    if B_old != None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / D.sum()

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "weight_for_bond": W_b,
        "validator_bond": B,
        "validator_ema_bond": B_ema,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "bond_alpha": bond_alpha,
        "alpha_a": a,
        "alpha_b": b
    }

def Yuma2(W, S, B_ema=None, kappa=0.5, bond_penalty=1, bond_alpha=0.1, liquid_alpha = False, alpha_high = 0.9, alpha_low = 0.7, precision = 100000, override_consensus_high = None, override_consensus_low = None):
    # === Weight === 
    W = (W.T / (W.sum(dim=1) +  1e-6)).T
    
    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1
        c_low = 0
        
        while (c_high - c_low) > 1/precision:
            c_mid = (c_high + c_low) / 2

            _c_sum = (miner_weight > c_mid) * S
            if sum(_c_sum) > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65535).int() / 65535

    # === Consensus cliped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = (W_clipped).sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - bond_penalty) * W + bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = None
    if liquid_alpha:
        if override_consensus_high == None:
            consensus_high = C.quantile(0.75)
        else:
            consensus_high = override_consensus_high 
        
        if override_consensus_low == None:
            consensus_low = C.quantile(0.25)
        else:
            consensus_low = override_consensus_low 

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)
        a = (math.log(1/alpha_high - 1) - math.log(1/ alpha_low - 1) ) / (consensus_low - consensus_high) 
        b = math.log(1/ alpha_low - 1) + a * consensus_low 
        alpha = 1 / (1 + math.e **(-a *C + b)) # alpha to the old weight
        bond_alpha = 1 - torch.clip(alpha, alpha_low, alpha_high)

    if B_ema == None:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D = D.nan_to_num(0)
    D_normalized = D / (D.sum() + 1e-6)  # Add epsilon to avoid division by zero


    # === EMA Bond for next epoch ===
    B_ema_next = bond_alpha * B + (1 - bond_alpha) * B_ema

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "server_trust": T,
        "validator_trust": T_v,
        "weight_for_bond": W_b,
        "validator_bond": B,
        "validator_ema_bond": B_ema_next,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized,
        "bond_alpha": bond_alpha,
        "alpha_a": a,
        "alpha_b": b
    }

def run_simulation(validators, stakes, stakes_units, weights_epochs, num_epochs, total_emission, emission_ratio=0.41, yuma_function=Yuma):
    dividends_per_validator = {validator: [] for validator in validators}
    bonds_per_epoch = []
    B_state = None  # Holds B_old or B_ema depending on the Yuma function

    for epoch in range(num_epochs):
        W = weights_epochs[epoch]
        S = stakes

        # Call the appropriate Yuma function
        if yuma_function == Yuma:
            result = yuma_function(W, S, B_old=B_state, kappa=0.5)
            B_state = result['validator_ema_bond']
        elif yuma_function == Yuma2:
            result = yuma_function(W, S, B_ema=B_state, kappa=0.5)
            B_state = result['validator_ema_bond']
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized = result['validator_reward_normalized']

        E_i = emission_ratio * D_normalized
        validator_emission = E_i * total_emission

        for i, validator in enumerate(validators):
            stake_unit = stakes_units[i]
            dividend_per_1000_tao = validator_emission[i].item() / stake_unit
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(result['validator_bond'].clone())

    return dividends_per_validator, bonds_per_epoch


def plot_results(num_epochs, dividends_per_validator, case):
    plt.close('all')  # Close all open figures
    plt.figure(figsize=(14, 6))

    num_epochs_calculated = None
    line_styles = ['-', '--', ':', '-.']
    x = None

    for validator, dividends in dividends_per_validator.items():
        # Convert dividends to numpy arrays
        if isinstance(dividends, torch.Tensor):
            dividends = dividends.detach().cpu().numpy()
        elif isinstance(dividends, list):
            dividends = np.array([float(d) for d in dividends])
        else:
            dividends = np.array(dividends, dtype=float)

        # Calculate num_epochs and x only once
        if num_epochs_calculated is None:
            num_epochs_calculated = len(dividends)
            x = list(range(num_epochs_calculated))

        # Plot the dividends for each validator
        y = dividends
        plt.plot(x, y, marker='o', label=validator, alpha=0.7, linestyle=line_styles.pop(0))

    # Adjust tick locations and labels
    if num_epochs_calculated == 40:  # For 2-day span (40 epochs)
        tick_locs = [0, 1, 2] + list(range(5, 39, 5)) + [39]
    elif num_epochs_calculated == 80:  # For 4-day span (80 epochs)
        tick_locs = [0, 1, 2, 41, 42, 43] + list(range(10, 79, 10)) + [79]
    else:
        tick_locs = [0, 1, 2, num_epochs_calculated - 1]

    tick_labels = [i if i in tick_locs else '' for i in range(num_epochs_calculated)]

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Dividend per 1,000 Tao per Epoch')
    plt.title(f'Validators dividends Over Time (2 days span) {case}')
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks=x, labels=tick_labels, fontsize=8)
    plt.tight_layout()
    plt.show()



def plot_total_dividends(total_dividends, case, validators):
    dividends = [total_dividends[validator] for validator in validators]
    
    x = np.arange(len(validators))
    width = 0.3
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    colors = colors[:len(validators)]
    
    plt.figure(figsize=(8, 4))
    bars = plt.bar(x, dividends, width, color=colors)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f'{height:.6f}', 
            ha='center', 
            va='bottom', 
            fontsize=9
        )
    
    plt.ylabel('Total Dividends per 1,000 Tao Staked')
    plt.title(f'Total Dividends per Validator per 1,000 Tao Staked\n{case}')
    plt.xticks(x, validators)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_weights(num_epochs, validators, servers, weights_per_epoch, case_name):
    x = list(range(num_epochs))
    plt.figure(figsize=(14, 6))

    for idx_v, validator in enumerate(validators):
        for idx_s, server in enumerate(servers):
            weights = []
            for epoch in range(num_epochs):
                W = weights_per_epoch[epoch]  # Shape: (num_validators, num_servers)
                weight = W[idx_v, idx_s].item()
                weights.append(weight)
            plt.plot(x, weights, label=f'{validator} to {server}')

    plt.xlabel('Epoch')
    plt.ylabel('Weight Assigned')
    plt.title(f'Weights Assigned by Validators per Server Over Time\n{case_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bonds(num_epochs, validators, servers, bonds_per_epoch, case_name):
    x = list(range(num_epochs))
    
    # Create a figure with two horizontally aligned subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(f'Validators bonds per Server Over Time\n{case_name}', fontsize=14)

    line_styles = ['-', '--', ':']
    style_cycle = itertools.cycle(line_styles)
    validator_styles = {validator: next(style_cycle) for validator in validators}

    # To collect handles for the shared legend
    handles = []
    labels = []
    
    for idx_s, server in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            bonds = []
            for epoch in range(num_epochs):
                B = bonds_per_epoch[epoch]
                bond = B[idx_v, idx_s].item()
                bonds.append(bond)

            linestyle = validator_styles[validator]
            line, = ax.plot(
                x, bonds, 
                marker='o', 
                alpha=0.7, 
                linestyle=linestyle,
                linewidth=2  # Optional: Adjust line width for better visibility
            )
            if idx_s == 0:
                # Only add labels for the first subplot to create a shared legend
                handles.append(line)
                labels.append(validator)
        
        # Configure each subplot for one server
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Bond Value' if idx_s == 0 else '')
        ax.set_title(f'{server}')
        ax.grid(True)
    
    # Create a single shared legend below the subplots
    fig.legend(handles, labels, loc='lower center', ncol=len(validators), bbox_to_anchor=(0.5, 0.02))
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.95])  # [left, bottom, right, top]
    plt.show()

def calculate_total_dividends(validators, dividends_per_validator, num_epochs=30):
    total_dividends = {}
    for validator in validators:
        if len(dividends_per_validator[validator]) >= num_epochs:
            dividends = dividends_per_validator[validator][:num_epochs]
        else:
            dividends = dividends_per_validator[validator]
        total_dividend = sum(dividends)
        total_dividends[validator] = total_dividend
    return total_dividends


def generate_weight_table(case_name, weights_epochs, validators, num_epochs):
    # Define which epochs to include based on total_epochs
    if num_epochs == 40:
        include_epochs = [1, 2, 3, 40]
    elif num_epochs == 80:
        include_epochs = [1, 2, 3, 41, 42, 43, 80]
    else:
        # For other epoch counts, adjust as needed
        if num_epochs > 7:
            include_epochs = list(range(1, 4)) + list(range(num_epochs - 2, num_epochs + 1))
        else:
            include_epochs = list(range(1, num_epochs + 1))

    include_set = set(include_epochs)
    data = []
    previous_weights = None

    for epoch_idx in range(1, num_epochs + 1):
        if epoch_idx in include_set:
            W = weights_epochs[epoch_idx - 1]
            current_weights = {validator: [round(w, 6) for w in W[idx_v].tolist()]
                               for idx_v, validator in enumerate(validators)}
            epoch_entry = {'Epoch': epoch_idx}
            epoch_entry.update(current_weights)
            data.append(epoch_entry)
            previous_weights = current_weights
        else:
            # Insert ellipsis row with previous_weights if not already inserted
            if not data or data[-1]['Epoch'] != '...':
                if previous_weights is not None:
                    ellipsis_entry = {'Epoch': '...'}
                    for validator in validators:
                        ellipsis_entry[validator] = previous_weights[validator]
                    data.append(ellipsis_entry)

    df = pd.DataFrame(data)
    df.set_index('Epoch', inplace=True)
    case_title = f"### {case_name} Weights per Epoch"
    display(Markdown(case_title))
    display(df)

    return df

total_emission = 100 # We assume this is the total emission in tao per epoch for the subnet

stakes = torch.tensor([0.8, 0.1, 0.1])
total_stake_tao = 1_000_000  # Total stake in the network
stakes_tao = stakes * total_stake_tao  # [800,000, 100,000, 100,000]
stakes_units = stakes_tao / 1_000  # [800, 100, 100]

validators = ['Big vali.', 'Small vali.', 'Small vali 2.']

dividends_per_validator, bonds_per_epoch = run_simulation(
    validators,
    stakes,
    stakes_units,
    weights_epochs_cases['Case 7'],
    40,
    total_emission,
    yuma_function=Yuma
)

print("Dividends per validator per 1,000 Tao staked:")