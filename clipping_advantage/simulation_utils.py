import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import base64
import io
from IPython.display import Markdown, display, HTML
from cases import cases


def YumaRust(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    bond_alpha: float = 0.1,
    liquid_alpha: bool = False,
    alpha_high: float = 0.9,
    alpha_low: float = 0.7,
    precision: int = 100_000,
    override_consensus_high: Optional[float] = None,
    override_consensus_low: Optional[float] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Enhanced Yuma function with operations order aligned to Rust's epoch function.
    Assumes a simulated environment with valid weights and active validators.
    """
    dtype = torch.float64  # Use float64 for higher precision

    # === Stake Normalization ===
    S_normalized = S / S.sum()

    # === Weight Normalization ===
    W_normalized = W / (W.sum(dim=1, keepdim=True) + 1e-6)

    # === Prerank Calculation ===
    P = (S_normalized.unsqueeze(1) * W_normalized).sum(dim=0)

    # === Consensus Calculation ===
    num_miners = W_normalized.shape[1]
    C = torch.zeros(num_miners, dtype=dtype)

    for i in range(num_miners):
        miner_weight = W_normalized[:, i]
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > (1 / precision):
            c_mid = (c_high + c_low) / 2.0
            _c_sum = torch.where(miner_weight > c_mid, S_normalized, torch.tensor(0.0, dtype=dtype)).sum()

            if _c_sum > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    # Normalize C to maintain precision similar to fixed-point
    C = (C / C.sum() * 65_535).round() / 65_535

    # === Consensus Clipped Weight ===
    W_clipped = torch.min(W_normalized, C)

    # === Ranks, Trust, Incentive ===
    # Compute ranks: R = SUM(i) w_ij * s_i
    R = (S_normalized.unsqueeze(1) * W_clipped).sum(dim=0)

    # Compute incentive: normalize ranks
    R_sum = R.sum()
    I = R / (R_sum + 1e-6)

    # Compute trust: T = R / P
    T = R / (P + 1e-6)

    # Validator Trust: sum of clipped weights per validator / sum of normalized weights
    W_normalized_sum = W_normalized.sum(dim=1)
    W_clipped_sum = W_clipped.sum(dim=1)
    T_v = W_clipped_sum / (W_normalized_sum + 1e-6)

    # === Bonds Calculation ===
    # Compute Bonds: B = S_normalized * W_clipped
    B = S_normalized.unsqueeze(1) * W_clipped
    B_sum = B.sum(dim=0)
    B = B / (B_sum + 1e-6)
    B = torch.nan_to_num(B)

    # === EMA Bonds Calculation ===
    if liquid_alpha:
        consensus_high = override_consensus_high if override_consensus_high is not None else torch.quantile(C, 0.75)
        consensus_low = override_consensus_low if override_consensus_low is not None else torch.quantile(C, 0.25)

        if consensus_high == consensus_low:
            consensus_high = torch.quantile(C, 0.99)

        if consensus_high > consensus_low and consensus_high != 0 and consensus_low < 0:
            # Calculate logistic function parameters
            a = (math.log(1 / alpha_high - 1) - math.log(1 / alpha_low - 1)) / (consensus_low - consensus_high)
            b = math.log(1 / alpha_low - 1) + a * consensus_low
            alpha_dynamic = 1 / (1 + torch.exp(-(a * C + b)))
            bond_alpha_adjusted = 1 - torch.clamp(alpha_dynamic, min=alpha_low, max=alpha_high)
        else:
            bond_alpha_adjusted = bond_alpha
    else:
        # === Normal EMA Bonds Calculation ===
        bond_alpha_adjusted = bond_alpha

    # Compute EMA Bonds
    if B_old is not None:
        B_ema = mat_ema_sparse(B, B_old, bond_alpha_adjusted)
    else:
        B_ema = B.clone()

    # === Normalize EMA Bonds ===
    B_ema_sum = B_ema.sum(dim=0)
    B_ema = B_ema / (B_ema_sum + 1e-6)
    B_ema = torch.nan_to_num(B_ema)

    # === Dividend Calculation ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    # === Emission Calculation ===
    emission_sum_I = I.sum() + 1e-6
    normalized_server_emission = I / emission_sum_I

    emission_sum_D = D_normalized.sum() + 1e-6
    normalized_validator_emission = D_normalized / emission_sum_D

    pruning_scores = normalized_validator_emission.clone()

    # === Output Compilation ===
    emissions = {
        "weight": W_normalized,
        "stake": S_normalized,
        "prerank": P,
        "consensus": C,
        "clipped_weight": W_clipped,
        "rank": R,
        "incentive": I,
        "trust": T,
        "validator_trust": T_v,
        "bond": B,
        "validator_ema_bond": B_ema,
        "dividend": D,
        "dividend_normalized": D_normalized,
        "bond_alpha": bond_alpha_adjusted,
        "server_emission": normalized_server_emission,
        "validator_reward_normalized": normalized_validator_emission,
        "pruning_scores": pruning_scores
    }

    return emissions

def Yuma(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    bond_penalty: float = 1.0,
    bond_alpha: float = 0.1,
    liquid_alpha: bool = False,
    alpha_high: float = 0.9,
    alpha_low: float = 0.7,
    precision: int = 100_000,
    override_consensus_high: Optional[float] = None,
    override_consensus_low: Optional[float] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Original Yuma function with bonds and EMA calculation.
    """
    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - bond_penalty) * W + bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = None
    if liquid_alpha:
        consensus_high = override_consensus_high if override_consensus_high is not None else C.quantile(0.75)
        consensus_low = override_consensus_low if override_consensus_low is not None else C.quantile(0.25)

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (math.log(1 / alpha_high - 1) - math.log(1 / alpha_low - 1)) / (consensus_low - consensus_high)
        b = math.log(1 / alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, alpha_low, alpha_high)

    if B_old is not None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

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

def Yuma2(
    W: torch.Tensor,
    S: torch.Tensor,
    B_ema: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    bond_penalty: float = 1.0,
    bond_alpha: float = 0.1,
    liquid_alpha: bool = False,
    alpha_high: float = 0.9,
    alpha_low: float = 0.7,
    precision: int = 100_000,
    override_consensus_high: Optional[float] = None,
    override_consensus_low: Optional[float] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Alternative Yuma function with different EMA bond calculation.
    """
    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - bond_penalty) * W + bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = None
    if liquid_alpha:
        consensus_high = override_consensus_high if override_consensus_high is not None else C.quantile(0.75)
        consensus_low = override_consensus_low if override_consensus_low is not None else C.quantile(0.25)

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (math.log(1 / alpha_high - 1) - math.log(1 / alpha_low - 1)) / (consensus_low - consensus_high)
        b = math.log(1 / alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, alpha_low, alpha_high)

    if B_ema is None:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D = D.nan_to_num(0)
    D_normalized = D / (D.sum() + 1e-6)

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

def Yuma21(
    W: torch.Tensor,
    W_prev: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    bond_penalty: float = 1.0,
    bond_alpha: float = 0.1,
    liquid_alpha: bool = False,
    alpha_high: float = 0.9,
    alpha_low: float = 0.7,
    precision: int = 100_000,
    override_consensus_high: Optional[float] = None,
    override_consensus_low: Optional[float] = None
) -> Dict[str, Union[torch.Tensor, float]]:
    """
    Original Yuma function with bonds and EMA calculation.
    """
    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T

    if W_prev is None:
        W_prev = W
        
    # === Stake ===
    S = S / S.sum()

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Trusts ===
    T = (R / P).nan_to_num(0)
    T_v = W_clipped.sum(dim=1) / W.sum(dim=1)

    # === Bonds ===
    W_b = (1 - bond_penalty) * W_prev + bond_penalty * W_clipped
    B = S.view(-1, 1) * W_b / (S.view(-1, 1) * W_b).sum(dim=0)
    B = B.nan_to_num(0)

    a = b = None
    if liquid_alpha:
        consensus_high = override_consensus_high if override_consensus_high is not None else C.quantile(0.75)
        consensus_low = override_consensus_low if override_consensus_low is not None else C.quantile(0.25)

        if consensus_high == consensus_low:
            consensus_high = C.quantile(0.99)

        a = (math.log(1 / alpha_high - 1) - math.log(1 / alpha_low - 1)) / (consensus_low - consensus_high)
        b = math.log(1 / alpha_low - 1) + a * consensus_low
        alpha = 1 / (1 + math.e ** (-a * C + b))  # alpha to the old weight
        bond_alpha = 1 - torch.clamp(alpha, alpha_low, alpha_high)

    if B_old is not None:
        B_ema = bond_alpha * B + (1 - bond_alpha) * B_old
    else:
        B_ema = B

    # === Dividend ===
    D = (B_ema * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

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

def Yuma3(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    decay_rate: float = 0.1,
    alpha: float = 0.1,
    maxint: int = 2 ** 64 - 1,
    precision: int = 100_000
) -> Dict[str, torch.Tensor]:
    """
    Yuma function with per-bond capacity constraints and decay.
    """
    # === Weight ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T  # Normalize weights per validator

    # === Stake ===
    S = S / S.sum()  # Normalize stakes

    # === Prerank ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus ===
    C = torch.zeros(W.shape[1])  # Shape: (num_miners,)

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = (miner_weight > c_mid) * S
            if _c_sum.sum() > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus clipped weight ===
    W_clipped = torch.min(W, C)

    # === Rank ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive ===
    I = (R / R.sum()).nan_to_num(0)

    # === Bonds ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    decay = 1 - decay_rate  # Decay factor

    # **Step 1: Compute Validator Capacities**
    capacity = S * maxint  # Validators' bond capacity based on their stake

    # **Step 3: Compute Remaining Capacity**
    capacity_per_bond = S.unsqueeze(1) * maxint
    remaining_capacity = capacity_per_bond - B_old
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # **Step 4: Compute Purchase Capacity**
    alpha_capacity = (alpha * capacity).unsqueeze(1)
    purchase_capacity = torch.min(alpha_capacity, remaining_capacity)  # Purchase capacity per validator for this epoch

    # **Step 5: Allocate Purchase to Miners**
    purchase = purchase_capacity * W  # Bonds purchase per validator per miner

    # **Step 6: Update Bonds with Decay and Purchase**
    B = decay * B_old + purchase
    B = torch.min(B, capacity_per_bond)  # Enforce capacity constraints

    # === Validator reward ===
    D = (B * I).sum(dim=1)
    D_normalized = D / (D.sum() + 1e-6)

    return {
        "weight": W,
        "stake": S,
        "server_prerank": P,
        "server_consensus_weight": C,
        "consensus_clipped_weight": W_clipped,
        "server_rank": R,
        "server_incentive": I,
        "validator_bonds": B,
        "validator_reward": D,
        "validator_reward_normalized": D_normalized
    }

def mat_ema_sparse(bonds_delta: torch.Tensor, bonds: torch.Tensor, alpha: float) -> torch.Tensor:
    return alpha * bonds_delta + (1 - alpha) * bonds

def run_simulation(
    validators: List[str],
    stakes: List[torch.Tensor],
    weights: List[torch.Tensor],
    num_epochs: int,
    total_emission: float,
    emission_ratio: float = 0.41,
    total_stake_tao: float = 1_000_000,
    yuma_function: Callable = Yuma,
    bond_penalty: float = 1.0,
) -> Tuple[Dict[str, List[float]], List[torch.Tensor]]:
    """
    Runs the simulation over multiple epochs using the specified Yuma function.
    """
    dividends_per_validator = {validator: [] for validator in validators}
    bonds_per_epoch = []
    B_state: Optional[torch.Tensor] = None  # Holds B_old or B_ema depending on the Yuma function
    prev_W: Optional[torch.Tensor] = None

    for epoch in range(num_epochs):
        W = weights[epoch]
        S = stakes[epoch]

        stakes_tao = S * total_stake_tao
        stakes_units = stakes_tao / 1_000

        # Call the appropriate Yuma function
        if yuma_function == Yuma:
            result = yuma_function(W=W, S=S, B_old=B_state, kappa=0.5, bond_penalty=bond_penalty)
            B_state = result['validator_ema_bond']
        elif yuma_function == Yuma21:
            result = yuma_function(W=W, prev_W=prev_W, S=S, B_ema=B_state, kappa=0.5, bond_penalty=bond_penalty)
            B_state = result['validator_ema_bond']
            prev_W = result['weight']
        elif yuma_function == Yuma3:
            result = yuma_function(W, S, B_old=B_state, decay_rate=0.1, alpha=0.1)
            B_state = result['validator_bonds']
        elif yuma_function == YumaRust:
            result = yuma_function(W, S, B_old=B_state, kappa=0.5)
            B_state = result['validator_ema_bond']
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized = result['validator_reward_normalized']

        E_i = emission_ratio * D_normalized
        validator_emission = E_i * total_emission

        for i, validator in enumerate(validators):
            stake_unit = stakes_units[i].item()
            validator_emission_i = validator_emission[i].item()
            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0  # No stake means no dividend per 1000 Tao
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(B_state.clone())

    return dividends_per_validator, bonds_per_epoch

def plot_to_base64():
    """
    Captures the current Matplotlib figure and encodes it as a Base64 string.
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)  # Use a higher DPI for sharper images
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()
    # Use `max-width` instead of fixed width for responsiveness
    return f'<img src="data:image/png;base64,{encoded_image}" style="max-width:1200px; height:auto;">'

def generate_chart_table(cases, yuma_versions, total_emission, total_stake_tao, servers, bond_penalty=1.0):
    """
    Generates an HTML table with embedded charts for all cases, Yuma versions, and all chart types.
    """
    # Initialize the table structure
    table_data = {"Case/Chart Name": []}
    for _, yuma_version in yuma_versions:
        table_data[yuma_version] = []

    def process_chart(chart_type, case_name, analysis, table_data, chart_base64_dict):
        row_content = f"{case_name} - {chart_type.capitalize()}<br><small>{analysis.get(chart_type, 'No analysis available.')}</small>"
        if row_content not in table_data["Case/Chart Name"]:
            table_data["Case/Chart Name"].append(row_content)
        for yuma_version, chart_base64 in chart_base64_dict.items():
            table_data[yuma_version].append(chart_base64)

    for case in cases:
        case_name = case['name']
        num_epochs = case['num_epochs']
        weights_epochs = case['weights_epochs']
        stakes = case['stakes_epochs']
        validators = case['validators']
        analysis = case['analysis']

        for chart_type in ['weights', 'dividends', 'bonds']:
            chart_base64_dict = {}
            for yuma_function, yuma_version in yuma_versions:
                full_case_name = f"{case_name} - {yuma_version}"
                if yuma_version in ["Yuma 1 (paper)", "Yuma 2 (Adrian-Fish)"]:
                    full_case_name = f"{full_case_name} - beta={bond_penalty}"

                # Run simulation to get dividends and bonds
                dividends_per_validator, bonds_per_epoch = run_simulation(
                    validators=validators,
                    stakes=stakes,
                    weights=weights_epochs,
                    num_epochs=num_epochs,
                    total_emission=total_emission,
                    total_stake_tao=total_stake_tao,
                    yuma_function=yuma_function,
                    bond_penalty=bond_penalty
                )

                # Generate the appropriate chart
                if chart_type == 'weights':
                    chart_base64 = plot_validator_server_weights(
                        validators=validators,
                        weights_epochs=weights_epochs,
                        servers=servers,
                        num_epochs=num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == 'dividends':
                    total_dividends = calculate_total_dividends(validators, dividends_per_validator, num_epochs)
                    chart_base64 = plot_results_2(
                        num_epochs=num_epochs,
                        dividends_per_validator=dividends_per_validator,
                        case=full_case_name,
                        total_dividends=total_dividends,
                        to_base64=True
                    )
                elif chart_type == 'bonds':
                    chart_base64 = plot_bonds(
                        num_epochs=num_epochs,
                        validators=validators,
                        servers=servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True
                    )

                chart_base64_dict[yuma_version] = chart_base64

            # Process the chart data for all yuma_versions
            process_chart(chart_type, case_name, analysis, table_data, chart_base64_dict)

    # Convert the table to a DataFrame
    summary_table = pd.DataFrame(table_data)

    # The rest of your code remains the same...

    # Add custom CSS for scrollable table
    custom_css = """
        <style>
            .scrollable-table-container {
                width: 100%;  /* Full-width container */
                overflow-x: auto;  /* Enable horizontal scrolling */
                overflow-y: hidden;  /* Prevent vertical scrolling on the container */
                white-space: nowrap;  /* Prevent wrapping of table content */
                border: 1px solid #ccc;  /* Optional border for better visualization */
            }
            table {
                border-collapse: collapse;
                table-layout: auto;  /* Allow dynamic column resizing */
                width: auto;  /* Let table width grow as needed */
            }
            td, th {
                padding: 10px;
                vertical-align: top;
                text-align: center;
            }
            img {
                display: block;
                margin: 0 auto;
                max-width: 100%;  /* Keep image size within its cell */
                height: auto;  /* Maintain aspect ratio */
            }
        </style>
        """

    # Convert the DataFrame to an HTML table with custom styles
    html_table = summary_table.to_html(escape=False, index=False)

    # Wrap the table in a scrollable div
    scrollable_table = f"""
    <div class="scrollable-table-container">
        {html_table}
    </div>
    """

    # Combine the CSS and scrollable table HTML
    full_html = custom_css + scrollable_table

    return HTML(full_html)

def plot_results(num_epochs: int, dividends_per_validator: Dict[str, List[float]], case: str, total_dividends: Dict[str, float], to_base64: bool = False):
    plt.close('all')  # Close all open figures
    fig, ax_main = plt.subplots(figsize=(14, 6))

    num_epochs_calculated = None
    marker_styles = ['+', 'x', '*', 'd']
    marker_sizes = [14, 10, 8, 8]
    line_styles = ['-', '--', ':', '-.']
    x = None

    for idx, (validator, dividends) in enumerate(dividends_per_validator.items()):
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
            x = np.array(range(num_epochs_calculated))

        delta = 0.05  # Adjust this value as needed
        x_shifted = x + idx * delta

        marker = marker_styles.pop(0)
        ax_main.plot(
            x_shifted,
            dividends,
            marker=marker,
            markersize=marker_sizes.pop(0),
            label=validator,
            alpha=0.7,
            linestyle=line_styles.pop(0)
        )

    # Adjust tick locations and labels for the main chart
    tick_locs = list(range(0, num_epochs_calculated, 5))  # Every 5th epoch
    tick_labels = [str(i) for i in tick_locs]

    ax_main.set_xticks(tick_locs)
    ax_main.set_xticklabels(tick_labels, fontsize=8)  # Bring back x-axis ticks
    ax_main.set_xlabel('Time (Epochs)')
    ax_main.set_ylim(bottom=0)
    ax_main.set_xlabel('Time (Epochs)')
    ax_main.set_ylabel('Dividend per 1,000 Tao per Epoch')
    ax_main.set_title(f'{case}')
    ax_main.legend()
    ax_main.grid(True)

    if case.startswith("Case 4"):
        ax_main.set_ylim(0, 0.042)  # Set specific height for Case 4

    # Add a small subplot for the horizontal bar chart
    ax_bar = fig.add_axes([0.15, -0.15, 0.7, 0.1])  # Adjust height and placement for smaller bars
    validators = list(total_dividends.keys())
    total_values = list(total_dividends.values())
    bar_colors = ['blue', 'orange', 'green', 'red'][:len(validators)]  # Adjust for number of validators

    # Create horizontal bars
    bars = ax_bar.barh(
        validators,
        total_values,
        color=bar_colors,
        alpha=0.7,
    )

    # Add numerical values next to each bar
    for bar, value in zip(bars, total_values):
        ax_bar.text(
            bar.get_width() + 0.02,  # Position slightly to the right of the bar
            bar.get_y() + bar.get_height() / 2,  # Center vertically within the bar
            f"{value:.6f}",
            va='center',
            fontsize=8
        )

    ax_bar.set_xlabel("Total Dividends")
    ax_bar.set_xlim(0, max(total_values) * 1.1)  # Add padding for better visibility
    ax_bar.set_yticks(range(len(validators)))
    ax_bar.set_yticklabels(validators, fontsize=8)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)

    # Adjust the figure layout to ensure proper alignment
    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return plot_to_base64()
    
    plt.show()

def plot_results_2(num_epochs: int, dividends_per_validator: Dict[str, List[float]], case: str, total_dividends: Dict[str, float], to_base64: bool = False):
    plt.close('all')  # Close all open figures
    fig, ax_main = plt.subplots(figsize=(14, 6))

    num_epochs_calculated = None
    marker_styles = ['+', 'x', '*', 'd']
    marker_sizes = [14, 10, 8, 8]
    line_styles = ['-', '--', ':', '-.']
    x = None

    for idx, (validator, dividends) in enumerate(dividends_per_validator.items()):
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
            x = np.array(range(num_epochs_calculated))

        delta = 0.05  # Adjust this value as needed
        x_shifted = x + idx * delta

        marker = marker_styles.pop(0)
        ax_main.plot(
            x_shifted,
            dividends,
            marker=marker,
            markersize=marker_sizes.pop(0),
            label=f"{validator} (Total: {total_dividends[validator]:.6f})",
            alpha=0.7,
            linestyle=line_styles.pop(0)
        )

    # Adjust tick locations and labels for the main chart
    tick_locs = [0, 1, 2] + list(range(5, num_epochs_calculated, 5))  # Every 5th epoch
    tick_labels = [str(i) for i in tick_locs]

    ax_main.set_xticks(tick_locs)
    ax_main.set_xticklabels(tick_labels, fontsize=8)  # Bring back x-axis ticks
    ax_main.set_xlabel('Time (Epochs)')
    ax_main.set_ylim(bottom=0)
    ax_main.set_xlabel('Time (Epochs)')
    ax_main.set_ylabel('Dividend per 1,000 Tao per Epoch')
    ax_main.set_title(f'{case}')
    ax_main.legend()
    ax_main.grid(True)

    if case.startswith("Case 4"):
        ax_main.set_ylim(0, 0.042)  # Set specific height for Case 4

    plt.subplots_adjust(hspace=0.3)

    if to_base64:
        return plot_to_base64()
    
    plt.show()

def plot_total_dividends(total_dividends: Dict[str, float], case: str, validators: List[str], to_base64: bool = False):
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

    if to_base64:
        return plot_to_base64()
    plt.show()

def plot_bonds(
    num_epochs: int,
    validators: List[str],
    servers: List[str],
    bonds_per_epoch: List[torch.Tensor],
    case_name: str,
    to_base64: bool = False
):
    x = list(range(num_epochs))

    # Create a figure with two horizontally aligned subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    fig.suptitle(f'Validators bonds per Server\n{case_name}', fontsize=14)

    combined_styles = [
        ('-', '+', 12, 2),  # Line style, marker style, marker size, marker edge width
        ('--', 'x', 12, 1),
        (':', '*', 8, 1)
    ]

    # Assign styles to each validator
    validator_styles = {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }

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

            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

            line, = ax.plot(
                x, bonds,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
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

    if to_base64:
        return plot_to_base64()
    plt.show()

def plot_validator_server_weights(
    validators: List[str],
    weights_epochs: List[torch.Tensor],
    servers: List[str],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False,
):
    marker_styles = ['+', 'x', '*', 'd']
    marker_sizes = [14, 10, 8, 8]
    line_styles = ['-', '--', ':', '-.']

    plt.figure(figsize=(14, 1))
    for idx_v, validator in enumerate(validators):
        server_over_time = []
        for epoch in range(num_epochs):
            W = weights_epochs[epoch]
            validator_weights = W[idx_v]  # weights given by validator idx_v at epoch
            server_idx = torch.argmax(validator_weights).item()
            server_supported = server_idx + 1  # Add 1 so that servers are 1 and 2
            server_over_time.append(server_supported)
        plt.plot(
            range(num_epochs),
            server_over_time,
            label=validator,
            marker=marker_styles.pop(0),
            markersize=marker_sizes.pop(0),
            linestyle=line_styles.pop(0)
        )


    plt.xlabel('Epoch')
    plt.ylabel('Server Supported')
    plt.title(f'Validators Weights to Servers \n{case_name}')
    plt.yticks([1, 2], servers)
    plt.legend()
    plt.grid(True)

    tick_locs = [0, 1, 2] + list(range(5, num_epochs, 5))  # First 3 ticks and then every 5th epoch
    plt.xticks(tick_locs, [str(i) for i in tick_locs])

    if to_base64:
        return plot_to_base64()
    plt.show()

def calculate_total_dividends(
    validators: List[str],
    dividends_per_validator: Dict[str, List[float]],
    num_epochs: int = 30
) -> Dict[str, float]:
    total_dividends = {}
    for validator in validators:
        dividends = dividends_per_validator[validator][:num_epochs]
        total_dividend = sum(dividends)
        total_dividends[validator] = total_dividend
    return total_dividends

def generate_weights_table(
    case_name: str,
    weights_epochs: List[torch.Tensor],
    validators: List[str],
    num_epochs: int
) -> pd.DataFrame:
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
            current_weights = {
                validator: [round(w, 6) for w in W[idx_v].tolist()]
                for idx_v, validator in enumerate(validators)
            }
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