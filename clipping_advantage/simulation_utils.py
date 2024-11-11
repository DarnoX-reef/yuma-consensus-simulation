import torch
import matplotlib.pyplot as plt
import numpy as np  
import math

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

def run_simulation(validators, stakes, stakes_tao, weights_epochs, num_epochs, total_emission, emission_ratio=0.41, yuma_function=Yuma):
    dividends_per_validator = {validator: [] for validator in validators}
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
            stake_tao = stakes_tao[i]
            dividend_per_1000_tao = validator_emission[i].item() / stake_tao
            dividends_per_validator[validator].append(dividend_per_1000_tao)

    return dividends_per_validator


def plot_results(num_epochs, dividends_per_validator, case):
    x = list(range(num_epochs))
    plt.figure(figsize=(14, 6))

    for validator, dividends in dividends_per_validator.items():
        # Ensure dividends are converted to numpy arrays
        if isinstance(dividends, torch.Tensor):
            dividends = dividends.detach().cpu().numpy()
        elif isinstance(dividends, list):
            dividends = np.array([float(d) for d in dividends])
        else:
            dividends = np.array(dividends, dtype=float)

        y = dividends
        plt.plot(x, y, marker='o', label=validator)

    if num_epochs == 40:  # For 2-day span (40 epochs)
        tick_locs = [0, 1, 2] + list(range(5, 39, 5)) + [39]
    elif num_epochs == 80:  # For 4-day span (80 epochs)
        tick_locs = [0, 1, 2, 41, 42, 43] + list(range(10, 79, 10)) + [79]
    else:
        tick_locs = [0, 1, 2, num_epochs - 1]
    
    tick_labels = [i if i in tick_locs else '' for i in range(num_epochs)]

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Dividend per 1,000 Tao per Epoch')
    plt.title(f'Dividends per Validator Over Time (2 days span) {case}')
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
    
    plt.xlabel('Validator')
    plt.ylabel('Total Dividends per 1,000 Tao Staked')
    plt.title(f'Total Dividends per Validator per 1,000 Tao Staked\n{case}')
    plt.xticks(x, validators)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
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