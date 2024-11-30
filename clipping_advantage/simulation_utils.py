
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
    W_clipped = torch.min(W_prev, C)

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

def Yuma31(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    decay_rate: float = 0.1,
    alpha: float = 0.1,
    maxint: int = 2 ** 64 - 1,
    precision: int = 100_000,
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

def Yuma32(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    decay_rate: float = 0.1,
    alpha: float = 0.1,
    maxint: int = 2 ** 64 - 1,
    precision: int = 100_000,
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

def Yuma4(
    W: torch.Tensor,
    S: torch.Tensor,
    B_old: Optional[torch.Tensor] = None,
    kappa: float = 0.5,
    bond_alpha: float = 0.1,
    decay_rate: float = 0.1,
    precision: int = 100_000,
) -> Dict[str, torch.Tensor]:
    """
    Yuma4 function with per-bond cap of 1 and decay.
    Bonds are between 0 and 1 per validator-miner relation, where bonds can increase
    by at most bond_alpha per epoch towards the cap, and decay over time if validators stop buying bonds.
    Dividends are calculated by multiplying the validator's stake with their bonds.
    """
    # === Weight Normalization ===
    W = (W.T / (W.sum(dim=1) + 1e-6)).T  # Normalize weights per validator

    # === Stake Normalization ===
    S = S / S.sum()  # Normalize stakes

    # === Prerank Calculation ===
    P = (S.view(-1, 1) * W).sum(dim=0)

    # === Consensus Calculation ===
    C = torch.zeros(W.shape[1])  # Shape: (num_miners,)

    for i, miner_weight in enumerate(W.T):
        c_high = 1.0
        c_low = 0.0

        while (c_high - c_low) > 1 / precision:
            c_mid = (c_high + c_low) / 2.0
            _c_sum = ((miner_weight > c_mid).float() * S).sum()
            if _c_sum > kappa:
                c_low = c_mid
            else:
                c_high = c_mid

        C[i] = c_high

    # Optional: Quantize consensus weights
    C = (C / C.sum() * 65_535).int() / 65_535

    # === Consensus Clipped Weight ===
    W_clipped = torch.min(W, C)

    # === Rank Calculation ===
    R = (S.view(-1, 1) * W_clipped).sum(dim=0)

    # === Incentive Calculation ===
    I = (R / R.sum()).nan_to_num(0)

    # === Bonds Calculation ===
    if B_old is None:
        B_old = torch.zeros_like(W)

    decay_factor = 1 - decay_rate
    # Apply decay to old bonds
    B_decayed = B_old * decay_factor

    # Remaining capacity per bond is cap - B_decayed
    remaining_capacity = 1.0 - B_decayed
    remaining_capacity = torch.clamp(remaining_capacity, min=0.0)

    # Purchase increment per validator-miner pair
    # Each validator can increase bonds by at most bond_alpha per epoch towards the cap
    purchase_increment = bond_alpha * W  # Validators allocate their purchase across miners based on weights
    # Ensure that purchase does not exceed remaining capacity
    purchase = torch.min(purchase_increment, remaining_capacity)

    # Update bonds
    B = B_decayed + purchase
    # Ensure bonds do not exceed the cap of 1
    B = torch.clamp(B, max=1.0)

    # === Dividends Calculation ===
    # D_i = S_i * sum_j B_ij
    total_bonds_per_validator = (B * I).sum(dim=1)  # Sum over miners for each validator
    D = S * total_bonds_per_validator  # Element-wise multiplication

    # Normalize dividends
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
    liquid_alpha: bool = False,
    reset_bonds_epoch: Optional[int] = None,
    reset_bonds_miner_index: Optional[int] = None,
    kappa: float = 0.5,
) -> Tuple[Dict[str, List[float]], List[torch.Tensor]]:
    """
    Runs the simulation over multiple epochs using the specified Yuma function.
    """
    dividends_per_validator = {validator: [] for validator in validators}
    bonds_per_epoch = []
    B_state: Optional[torch.Tensor] = None
    W_prev: Optional[torch.Tensor] = None
    server_consensus_weight: Optional[torch.Tensor] = None


    for epoch in range(num_epochs):
        W = weights[epoch]
        S = stakes[epoch]

        stakes_tao = S * total_stake_tao
        stakes_units = stakes_tao / 1_000

        # Call the appropriate Yuma function
        if yuma_function == Yuma:
            result = yuma_function(W=W, S=S, B_old=B_state, kappa=kappa, bond_penalty=bond_penalty, liquid_alpha=liquid_alpha)
            B_state = result['validator_ema_bond']
        elif yuma_function == Yuma2:
            result = yuma_function(W=W, W_prev=W_prev, S=S, B_old=B_state, kappa=kappa, bond_penalty=bond_penalty)
            B_state = result['validator_ema_bond']
            W_prev = result['weight']
        elif yuma_function == Yuma3:
            result = yuma_function(W, S, B_old=B_state, kappa=kappa, decay_rate=0.1, alpha=0.1)
            B_state = result['validator_bonds']
        elif yuma_function == Yuma31:
            if B_state is not None and epoch == reset_bonds_epoch:
                B_state[:, reset_bonds_miner_index] = 0.0
            result = yuma_function(W, S, B_old=B_state, kappa=kappa, decay_rate=0.1, alpha=0.1)
            B_state = result['validator_bonds']
        elif yuma_function == Yuma32:
            if B_state is not None and epoch == reset_bonds_epoch and server_consensus_weight[reset_bonds_miner_index] == 0.0:
                B_state[:, reset_bonds_miner_index] = 0.0
            result = yuma_function(W, S, B_old=B_state, kappa=kappa, decay_rate=0.1, alpha=0.1)
            B_state = result['validator_bonds']
            server_consensus_weight = result['server_consensus_weight']
        elif yuma_function == Yuma4:
            if B_state is not None and epoch == reset_bonds_epoch and server_consensus_weight[reset_bonds_miner_index] == 0.0:
                B_state[:, reset_bonds_miner_index] = 0.0
            result = yuma_function(W, S, B_old=B_state, kappa=kappa, bond_alpha=0.1, decay_rate=0.1)
            B_state = result['validator_bonds']
            server_consensus_weight = result['server_consensus_weight']
        elif yuma_function == YumaRust:
            result = yuma_function(W, S, B_old=B_state, kappa=kappa)
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
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', dpi=150)  # Use a higher DPI for sharper images
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('ascii')
    buf.close()
    plt.close()
    # Use max-width instead of fixed width for responsiveness
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
                if yuma_version in ["Yuma 1 (paper)", "Yuma 1 (paper) - liquid alpha on", "Yuma 2 (Adrian-Fish)"]:
                    full_case_name = f"{full_case_name} - beta={bond_penalty}"

                liquid_alpha = False
                if yuma_version == "Yuma 1 (paper) - liquid alpha on":
                    liquid_alpha = True

                reset_bonds_epoch = None
                reset_bonds_miner_index = None
                if yuma_function == Yuma31 or yuma_function == Yuma32:
                    reset_bonds_epoch = 20
                    reset_bonds_miner_index = 1
                    
                # Run simulation to get dividends and bonds
                dividends_per_validator, bonds_per_epoch = run_simulation(
                    validators=validators,
                    stakes=stakes,
                    weights=weights_epochs,
                    num_epochs=num_epochs,
                    total_emission=total_emission,
                    total_stake_tao=total_stake_tao,
                    yuma_function=yuma_function,
                    bond_penalty=bond_penalty,
                    liquid_alpha=liquid_alpha,
                    reset_bonds_epoch=reset_bonds_epoch,
                    reset_bonds_miner_index=reset_bonds_miner_index
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
                    chart_base64 = plot_results(
                        num_epochs=num_epochs,
                        validators=validators,
                        dividends_per_validator=dividends_per_validator,
                        case=full_case_name,
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

    custom_css = """
<style>
    .scrollable-table-container {
        width: 100%; 
        overflow-x: auto;
        overflow-y: hidden;
        white-space: nowrap;
        border: 1px solid #ccc;  
        background-color: hsl(0, 0%, 98%) !important;  /* Light background for the table container */
    }
    table {
        border-collapse: collapse;
        table-layout: auto;
        width: auto;
        background-color: hsl(0, 0%, 100%) !important;  /* Pure white table background */
    }
    td, th {
        padding: 10px;
        vertical-align: top;
        text-align: center;
        color: hsl(0, 0%, 20%) !important;  /* Dark text for readability */
    }
    tr:nth-child(even) {
        background-color: hsl(0, 0%, 95%) !important;  /* Light gray for even rows */
    }
    tr:nth-child(odd) {
        background-color: hsl(0, 0%, 100%) !important;  /* Pure white for odd rows */
    }
</style>
"""


    html_table = summary_table.to_html(escape=False, index=False)
    scrollable_table = f"""
    <div class="scrollable-table-container">
        {html_table}
    </div>
    """

    full_html = custom_css + scrollable_table

    return HTML(full_html)

def plot_results(
        num_epochs: int,
        validators: list[str],
        dividends_per_validator: Dict[str, List[float]],
        case: str,
        to_base64: bool = False
        ):
    plt.close('all')  # Close all open figures
    fig, ax_main = plt.subplots(figsize=(14, 6))

    num_epochs_calculated = None
    x = None

    combined_styles = [
        ('-', '+', 12, 2),
        ('--', 'x', 12, 1),
        (':', 'o', 4, 1)
    ]

    validator_styles = {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }

    total_dividends, base_validator, percentage_diff_vs_base = calculate_total_dividends(
        validators,
        dividends_per_validator,
        num_epochs
    )

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

        linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

        total_dividend = total_dividends[validator]
        if validator == base_validator:
            label = f"(Base) {validator}: Total = {total_dividend:.6f}"
        else:
            percentage_diff = percentage_diff_vs_base[validator]
            label = f"{validator}: Total = {total_dividend:.6f} (+{percentage_diff:.1f}% vs. Base)"


        ax_main.plot(
            x_shifted,
            dividends,
            marker=marker,
            markeredgewidth=markeredgewidth,
            markersize=markersize,
            label=label,
            alpha=0.7,
            linestyle=linestyle
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
        (':', 'o', 4, 1)
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
) -> Tuple[Dict[str, float], str, Dict[str, float]]:
    """
    Calculates the total dividends per validator and computes the percentage difference
    relative to the validator with the lowest total dividend (the base).
    
    Returns:
        total_dividends: Dict mapping validator names to their total dividends.
        base_validator: The name of the validator with the lowest total dividend.
        percentage_diff_vs_base: Dict mapping validator names to their percentage difference vs. base.
    """
    total_dividends = {}
    for validator in validators:
        dividends = dividends_per_validator[validator][:num_epochs]
        total_dividend = sum(dividends)
        total_dividends[validator] = total_dividend

    # Identify the base validator (validator with the lowest total dividend)
    base_validator = min(total_dividends, key=total_dividends.get)
    base_dividend = total_dividends[base_validator]

    # Compute percentage difference vs base for each validator
    percentage_diff_vs_base = {}
    for validator, total_dividend in total_dividends.items():
        if validator == base_validator:
            percentage_diff_vs_base[validator] = 0.0  # Base validator has 0% difference vs itself
        else:
            percentage_diff = ((total_dividend - base_dividend) / base_dividend) * 100.0
            percentage_diff_vs_base[validator] = percentage_diff

    return total_dividends, base_validator, percentage_diff_vs_base

servers = ['Server 1', 'Server 2']

total_emission = 100 # We assume this is the total emission in tao per epoch for the subnet
total_stake_tao = 1_000_000  # Total stake in the network

case_name = cases[9]['name']
num_epochs = cases[9]['num_epochs']
weights_epochs = cases[9]['weights_epochs']
stakes = cases[9]['stakes_epochs']
validators = cases[9]['validators']

dividends_per_validator, bonds_per_epoch = run_simulation(
    validators=validators,
    stakes=stakes,
    weights=weights_epochs,
    num_epochs=num_epochs,
    total_emission=total_emission,
    total_stake_tao=total_stake_tao,
    yuma_function=Yuma2
)
