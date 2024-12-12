import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import base64
import io
from IPython.display import HTML
from typing import Dict, List, Optional, Tuple
from cases import cases, BaseCase
from yumas import (
    YumaRust,
    Yuma,
    Yuma2,
    Yuma3,
    Yuma4,
    YumaConfig,
    YumaSimulationNames,
    YumaParams,
    SimulationHyperparameters,
)

def run_simulation(
    case: BaseCase,
    yuma_version: str,
    yuma_config: YumaConfig,
) -> Tuple[Dict[str, List[float]], List[torch.Tensor]]:
    """
    Runs the simulation over multiple epochs using the specified Yuma function.
    """
    dividends_per_validator = {validator: [] for validator in case.validators}
    bonds_per_epoch = []
    server_incentives_per_epoch = []
    B_state: Optional[torch.Tensor] = None
    W_prev: Optional[torch.Tensor] = None
    server_consensus_weight: Optional[torch.Tensor] = None

    simulation_names = YumaSimulationNames()


    for epoch in range(case.num_epochs):
        W = case.weights_epochs[epoch]
        S = case.stakes_epochs[epoch]

        stakes_tao = S * yuma_config.total_subnet_stake
        stakes_units = stakes_tao / 1_000

        # Call the appropriate Yuma function
        if yuma_version in [simulation_names.YUMA, simulation_names.YUMA_LIQUID]:
            result = Yuma(W=W, S=S, B_old=B_state, config=yuma_config)
            B_state = result['validator_ema_bond']
        elif yuma_version == simulation_names.YUMA2:
            result = Yuma2(W=W, W_prev=W_prev, S=S, B_old=B_state, config=yuma_config)
            B_state = result['validator_ema_bond']
            W_prev = result['weight']
        elif yuma_version == simulation_names.YUMA3:
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result['validator_bonds']
        elif yuma_version == simulation_names.YUMA31:
            if B_state is not None and epoch == case.reset_bonds_epoch:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result['validator_bonds']
        elif yuma_version == simulation_names.YUMA32:
            if B_state is not None and epoch == case.reset_bonds_epoch and server_consensus_weight[case.reset_bonds_index] == 0.0:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma3(W, S, B_old=B_state, config=yuma_config)
            B_state = result['validator_bonds']
            server_consensus_weight = result['server_consensus_weight']
        elif yuma_version in [simulation_names.YUMA4, simulation_names.YUMA4_LIQUID]:
            if B_state is not None and epoch == case.reset_bonds_epoch and server_consensus_weight[case.reset_bonds_index] == 0.0:
                B_state[:, case.reset_bonds_index] = 0.0
            result = Yuma4(W, S, B_old=B_state, config=yuma_config)
            B_state = result['validator_bonds']
            server_consensus_weight = result['server_consensus_weight']
        elif yuma_version == "Yuma 0 (subtensor)":
            result = YumaRust(W, S, B_old=B_state, config=yuma_config)
            B_state = result['validator_ema_bond']
        else:
            raise ValueError("Invalid Yuma function.")

        D_normalized = result['validator_reward_normalized']

        E_i = yuma_config.validator_emission_ratio * D_normalized
        validator_emission = E_i * yuma_config.total_epoch_emission

        for i, validator in enumerate(case.validators):
            stake_unit = stakes_units[i].item()
            validator_emission_i = validator_emission[i].item()
            if stake_unit > 1e-6:
                dividend_per_1000_tao = validator_emission_i / stake_unit
            else:
                dividend_per_1000_tao = 0.0  # No stake means no dividend per 1000 Tao
            dividends_per_validator[validator].append(dividend_per_1000_tao)

        bonds_per_epoch.append(B_state.clone())
        server_incentives_per_epoch.append(result['server_incentive'])

    return dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch

def generate_chart_table(cases, yuma_versions, yuma_hyperparameters, draggable_table=False):
    """
    Generates an HTML table with embedded charts for all cases, Yuma versions, and all chart types.
    Applies alternating background colors for groups of three charts.
    """

    # Initialize the table structure
    table_data = {yuma_version: [] for yuma_version, _ in yuma_versions}

    def process_chart(table_data, chart_base64_dict):
        for yuma_version, chart_base64 in chart_base64_dict.items():
            content = f"{chart_base64}"
            table_data[yuma_version].append(content)

    for idx, case in enumerate(cases):
        chart_types = ['weights', 'dividends', 'bonds', 'normalized_bonds', 'incentives'] if idx in [9, 10] else ['weights', 'dividends', 'bonds', 'normalized_bonds']

        for chart_type in chart_types:
            chart_base64_dict = {}
            for yuma_version, yuma_params in yuma_versions:
                yuma_config = YumaConfig(
                    simulation=yuma_hyperparameters,
                    yuma_params=yuma_params,
                )
                full_case_name = f"{case.name} - {yuma_version}"
                if yuma_version in ["Yuma 1 (paper)", "Yuma 1 (paper) - liquid alpha on", "Yuma 2 (Adrian-Fish)"]:
                    full_case_name = f"{full_case_name} - beta={yuma_config.bond_penalty}"

                # Run simulation to get dividends and bonds
                dividends_per_validator, bonds_per_epoch, server_incentives_per_epoch = run_simulation(
                    case=case,
                    yuma_version=yuma_version,
                    yuma_config=yuma_config,
                )

                # Generate the appropriate chart
                if chart_type == 'weights':
                    chart_base64 = plot_validator_server_weights(
                        validators=case.validators,
                        weights_epochs=case.weights_epochs,
                        servers=case.servers,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True,
                    )
                elif chart_type == 'dividends':
                    chart_base64 = plot_results(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        dividends_per_validator=dividends_per_validator,
                        case=full_case_name,
                        base_validator=case.base_validator,
                        to_base64=True
                    )
                elif chart_type == 'bonds':
                    chart_base64 = plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True
                    )
                elif chart_type == 'normalized_bonds':
                    chart_base64 = plot_bonds(
                        num_epochs=case.num_epochs,
                        validators=case.validators,
                        servers=case.servers,
                        bonds_per_epoch=bonds_per_epoch,
                        case_name=full_case_name,
                        to_base64=True,
                        normalize=True
                    )
                elif chart_type == 'incentives':
                    chart_base64 = plot_incentives(
                        servers=case.servers,
                        server_incentives_per_epoch=server_incentives_per_epoch,
                        num_epochs=case.num_epochs,
                        case_name=full_case_name,
                        to_base64=True
                    )

                chart_base64_dict[yuma_version] = chart_base64

            # Process the chart data for all yuma_versions
            process_chart(table_data, chart_base64_dict)

 
    # Convert the table to a DataFrame
    summary_table = pd.DataFrame(table_data)

    if draggable_table:
        full_html = generate_draggable_html_table(table_data, summary_table)
    else:
        full_html = generate_ipynb_table(table_data, summary_table)

    return HTML(full_html)

def generate_draggable_html_table(table_data, summary_table):
    custom_css_js = """
    <style>
        body {
            margin: 0;
            padding: 0;
            overflow: hidden;
        }
        .scrollable-table-container {
            width: 100%; 
            height: 100vh; /* Full screen height */
            overflow: hidden; /* No traditional scrollbars */
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid #ccc;
            position: relative; 
            cursor: grab;
        }
        .scrollable-table-container:active {
            cursor: grabbing;
        }
        table {
            border-collapse: collapse;
            margin: 0 auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }
    </style>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const container = document.querySelector('.scrollable-table-container');
            let isDown = false;
            let startX, startY, scrollLeft, scrollTop;

            container.addEventListener('mousedown', (e) => {
                isDown = true;
                startX = e.pageX - container.offsetLeft;
                startY = e.pageY - container.offsetTop;
                scrollLeft = container.scrollLeft;
                scrollTop = container.scrollTop;
            });

            container.addEventListener('mouseleave', () => {
                isDown = false;
            });

            container.addEventListener('mouseup', () => {
                isDown = false;
            });

            container.addEventListener('mousemove', (e) => {
                if (!isDown) return;
                e.preventDefault();
                const x = e.pageX - container.offsetLeft;
                const y = e.pageY - container.offsetTop;
                const walkX = (x - startX) * 1; 
                const walkY = (y - startY) * 1; 
                container.scrollLeft = scrollLeft - walkX;
                container.scrollTop = scrollTop - walkY;
            });
        });
    </script>
    """
    
    # Generate HTML rows
    html_rows = []
    for i in range(len(next(iter(table_data.values())))):  # Number of rows
        row_html = '<tr>'
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f'<td>{cell_content}</td>'
        row_html += '</tr>'
        html_rows.append(row_html)

    # Combine rows and create the final table
    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{''.join(f'<th>{col}</th>' for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css_js + html_table

def generate_ipynb_table(table_data, summary_table):
    custom_css = """
    <style>
        .scrollable-table-container {
            width: 100%; 
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            border: 1px solid #ccc;  
            background-color: hsl(0, 0%, 98%);
        }
        table {
            border-collapse: collapse;
            table-layout: auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }
    </style>
    """

    # Generate HTML rows
    html_rows = []
    for i in range(len(next(iter(table_data.values())))):  # Number of rows
        row_html = '<tr>'
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f'<td>{cell_content}</td>'
        row_html += '</tr>'
        html_rows.append(row_html)

    # Combine rows and create the final table
    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{''.join(f'<th>{col}</th>' for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css + html_table

def plot_results(
    num_epochs: int,
    validators: list[str],
    dividends_per_validator: Dict[str, List[float]],
    case: str,
    base_validator: str,
    to_base64: bool = False
):

    plt.close('all')
    _, ax_main = plt.subplots(figsize=(14, 6))

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

    total_dividends, percentage_diff_vs_base = calculate_total_dividends(
        validators,
        dividends_per_validator,
        base_validator,
        num_epochs
    )

    for idx, (validator, dividends) in enumerate(dividends_per_validator.items()):
        if isinstance(dividends, torch.Tensor):
            dividends = dividends.detach().cpu().numpy()
        elif isinstance(dividends, list):
            dividends = np.array([float(d) for d in dividends])
        else:
            dividends = np.array(dividends, dtype=float)

        if num_epochs_calculated is None:
            num_epochs_calculated = len(dividends)
            x = np.array(range(num_epochs_calculated))

        delta = 0.05  # Adjust this value as needed
        x_shifted = x + idx * delta

        linestyle, marker, markersize, markeredgewidth = validator_styles[validator]

        total_dividend = total_dividends[validator]

        percentage_diff = percentage_diff_vs_base[validator]

        if percentage_diff > 0:
            percentage_str = f"(+{percentage_diff:.1f}%)"
        elif percentage_diff < 0:
            percentage_str = f"({percentage_diff:.1f}%)"
        else:
            percentage_str = "(Base)"

        label = f"{validator}: Total = {total_dividend:.6f} {percentage_str}"

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

    tick_locs = [0, 1, 2] + list(range(5, num_epochs_calculated, 5))  # Every 5th epoch
    tick_labels = [str(i) for i in tick_locs]

    ax_main.set_xticks(tick_locs)
    ax_main.set_xticklabels(tick_labels, fontsize=8)
    ax_main.set_xlabel('Time (Epochs)')
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel('Dividend per 1,000 Tao per Epoch')
    ax_main.set_title(f'{case}')
    ax_main.grid(True)
    ax_main.legend()

    # Glitch fix for Case 4
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
    to_base64: bool = False,
    normalize: bool = False
):
    """
    Plots bonds per server over epochs for each validator.
    If `normalize=True`, each epoch for a given server is normalized so that
    the sum of bonds across validators equals 1.
    """
    x = list(range(num_epochs))

    fig, axes = plt.subplots(1, len(servers), figsize=(14, 5), sharex=True, sharey=True)
    if len(servers) == 1:
        axes = [axes]  # Ensure axes is always a list
    if normalize:
        fig.suptitle(f'Validators bonds per Server normalized\n{case_name}', fontsize=14)
    else:
        fig.suptitle(f'Validators bonds per Server\n{case_name}', fontsize=14)

    combined_styles = [
        ('-', '+', 12, 2),   # (linestyle, marker, markersize, markeredgewidth)
        ('--', 'x', 12, 1),
        (':', 'o', 4, 1)
    ]

    validator_styles = {
        validator: combined_styles[idx % len(combined_styles)]
        for idx, validator in enumerate(validators)
    }

    # Step 1: Extract bonds data
    bonds_data = []
    for idx_s, _ in enumerate(servers):
        server_bonds = []
        for idx_v, _ in enumerate(validators):
            validator_bonds = []
            for epoch in range(num_epochs):
                B = bonds_per_epoch[epoch]
                bond_value = B[idx_v, idx_s].item()
                validator_bonds.append(bond_value)
            server_bonds.append(validator_bonds)
        bonds_data.append(server_bonds)

    # Step 2: Normalize if requested
    if normalize:
        for idx_s in range(len(servers)):
            for epoch in range(num_epochs):
                epoch_bonds = [bonds_data[idx_s][idx_v][epoch] for idx_v in range(len(validators))]
                total = sum(epoch_bonds)
                if total > 1e-12:
                    for idx_v in range(len(validators)):
                        bonds_data[idx_s][idx_v][epoch] /= total

    # Step 3: Plot the data
    handles, labels = [], []
    for idx_s, server in enumerate(servers):
        ax = axes[idx_s]
        for idx_v, validator in enumerate(validators):
            bonds = bonds_data[idx_s][idx_v]

            linestyle, marker, markersize, markeredgewidth = validator_styles[validator]
            line, = ax.plot(
                x, bonds,
                alpha=0.7,
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linestyle=linestyle,
                linewidth=2
            )

            if idx_s == 0:
                handles.append(line)
                labels.append(validator)

        ax.set_xlabel('Epoch')
        if normalize:
            ax.set_ylabel('Bond Ratio' if idx_s == 0 else '')
        else:
            ax.set_ylabel('Bond Value' if idx_s == 0 else '')
        ax.set_title(server)
        ax.grid(True)

        if normalize:
            ax.set_ylim(0, 1.05)

    fig.legend(handles, labels, loc='lower center', ncol=len(validators), bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=[0, 0.05, 0.98, 0.95])

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

    marker_styles = ['+', 'x', '*']
    line_styles = ['-', '--', ':']
    marker_sizes = [14, 10, 8]

    # Collect all y-values from weights_epochs
    y_values_all = [
        weights_epochs[epoch][idx_v][1].item()
        for idx_v in range(len(validators))
        for epoch in range(num_epochs)
    ]
    unique_y_values = sorted(set(y_values_all))

    # Define thresholds
    min_label_distance = 0.05
    close_to_server_threshold = 0.02

    # Function to determine if a value is a round number (e.g., multiples of 0.05)
    def is_round_number(y):
        return abs((y * 20) - round(y * 20)) < 1e-6  # Checks if value is a multiple of 0.05

    # Initialize y-ticks with server labels
    y_tick_positions = [0.0, 1.0]
    y_tick_labels = [servers[0], servers[1]]

    # Process other y-values
    for y in unique_y_values:
        if y in [0.0, 1.0]:
            continue  # Already added servers
        if abs(y - 0.0) < close_to_server_threshold or abs(y - 1.0) < close_to_server_threshold:
            continue  # Skip y-values too close to server labels
        # Check if y is a round number
        if is_round_number(y):
            # Check distance to existing y-ticks
            if all(abs(y - existing_y) >= min_label_distance for existing_y in y_tick_positions):
                y_tick_positions.append(y)
                # Format label as integer percentage if no decimal part, else one decimal
                y_percentage = y * 100
                label = f'{y_percentage:.0f}%'
                y_tick_labels.append(label)
        else:
            # For non-round numbers, only add if not too close
            if all(abs(y - existing_y) >= min_label_distance for existing_y in y_tick_positions):
                y_tick_positions.append(y)
                # Format label as percentage with one decimal place
                y_percentage = y * 100
                label = f'{y_percentage:.0f}%'
                y_tick_labels.append(label)

    # Sort y-ticks and labels together
    ticks = list(zip(y_tick_positions, y_tick_labels))
    ticks.sort()
    y_tick_positions, y_tick_labels = zip(*ticks)
    y_tick_positions = list(y_tick_positions)
    y_tick_labels = list(y_tick_labels)

    # Determine figure height based on the number of y-ticks
    fig_height = 1 if len(y_tick_positions) <= 2 else 3
    plt.figure(figsize=(14, fig_height))

    # Set y-limits
    plt.ylim(-0.05, 1.05)

    # Plot the data
    for idx_v, validator in enumerate(validators):
        y_values = [
            weights_epochs[epoch][idx_v][1].item() for epoch in range(num_epochs)
        ]
        plt.plot(
            range(num_epochs),
            y_values,
            label=validator,
            marker=marker_styles[idx_v % len(marker_styles)],
            linestyle=line_styles[idx_v % len(line_styles)],
            markersize=marker_sizes[idx_v % len(marker_sizes)],
            linewidth=2
        )

    plt.xlabel('Epoch')
    plt.title(f'Validators Weights to Servers \n{case_name}')

    # Set y-ticks and labels
    plt.yticks(y_tick_positions, y_tick_labels)
    plt.legend()
    plt.grid(True)

    # Set x-axis ticks
    tick_locs = list(range(0, num_epochs, 5))
    if 0 not in tick_locs:
        tick_locs.insert(0, 0)
    plt.xticks(tick_locs, [str(i) for i in tick_locs])

    if to_base64:
        return plot_to_base64()
    plt.show()

def plot_incentives(
    servers: List[str],
    server_incentives_per_epoch: List[torch.Tensor],
    num_epochs: int,
    case_name: str,
    to_base64: bool = False
):
    x = np.arange(num_epochs)
    plt.figure(figsize=(14, 3))

    for idx_s, server in enumerate(servers):
        incentives = [server_incentives[idx_s].item() for server_incentives in server_incentives_per_epoch]
        plt.plot(x, incentives, label=server)

    plt.xlabel('Epoch')
    plt.ylabel('Server Incentive')
    plt.title(f'Server Incentives\n{case_name}')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True)

    if to_base64:
        return plot_to_base64()
    plt.show()

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

def calculate_total_dividends(
    validators: List[str],
    dividends_per_validator: Dict[str, List[float]],
    base_validator: str,
    num_epochs: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Calculates the total dividends per validator and computes the percentage difference
    relative to the provided base validator.

    Returns:
        total_dividends: Dict mapping validator names to their total dividends.
        percentage_diff_vs_base: Dict mapping validator names to their percentage difference vs. base.
    """
    total_dividends = {}
    for validator in validators:
        dividends = dividends_per_validator.get(validator, [])
        dividends = dividends[:num_epochs]
        total_dividend = sum(dividends)
        total_dividends[validator] = total_dividend

    # Get base dividend
    base_dividend = total_dividends.get(base_validator, None)
    if base_dividend is None or base_dividend == 0.0:
        print(f"Warning: Base validator '{base_validator}' has zero or missing total dividends.")
        base_dividend = 1e-6  # Assign a small epsilon value to avoid division by zero

    # Compute percentage difference vs base for each validator
    percentage_diff_vs_base = {}
    for validator, total_dividend in total_dividends.items():
        if validator == base_validator:
            percentage_diff_vs_base[validator] = 0.0  # Base validator has 0% difference vs itself
        else:
            percentage_diff = ((total_dividend - base_dividend) / base_dividend) * 100.0
            percentage_diff_vs_base[validator] = percentage_diff

    return total_dividends, percentage_diff_vs_base

def generate_total_dividends_table(
    cases: List[BaseCase],
    yuma_versions: List[Tuple[str, YumaParams]],
    simulation_hyperparameters: SimulationHyperparameters,
) -> pd.DataFrame:
    """
    Generates a DataFrame with total dividends per standardized validator (A, B, C)
    for each Yuma version and case.
    """
    standardized_validators = ['Validator A', 'Validator B', 'Validator C']
    rows = []

    for case in cases:
        # Ensure exactly three validators
        if len(case.validators) != 3:
            raise ValueError(f"Case '{case.name}' does not have exactly 3 validators.")

        # Map original validators to standardized names
        validator_mapping = {
            case.validators[0]: 'Validator A',
            case.validators[1]: 'Validator B',
            case.validators[2]: 'Validator C',
        }

        # Initialize a row with 'Case'
        row = {'Case': case.name}

        for yuma_version, yuma_params in yuma_versions:
            # Create the Yuma configuration for the current version
            yuma_config = YumaConfig(
                simulation=simulation_hyperparameters,
                yuma_params=yuma_params,
            )

            # Run simulation
            dividends_per_validator, _, _ = run_simulation(
                case=case,
                yuma_version=yuma_version,
                yuma_config=yuma_config,
            )

            # Calculate total dividends
            total_dividends, _ = calculate_total_dividends(
                validators=case.validators,
                dividends_per_validator=dividends_per_validator,
                base_validator=case.base_validator,
                num_epochs=case.num_epochs
            )

            # Map dividends to standardized validator names
            standardized_dividends = {
                validator_mapping[orig_val]: total_dividends.get(orig_val, 0.0)
                for orig_val in case.validators
            }

            # Populate row with dividends for each Yuma version and validator
            for std_validator in standardized_validators:
                dividend = standardized_dividends.get(std_validator, 0.0)
                column_name = f"{std_validator} - {yuma_version}"
                row[column_name] = dividend

        # Append the populated row for the current case
        rows.append(row)

    # Create DataFrame from all rows
    df = pd.DataFrame(rows)

    # Optional: Reorder columns to have 'Case' first and then Yuma versions in the provided order
    columns = ['Case']
    for yuma_version, _ in yuma_versions:
        for std_validator in standardized_validators:
            col_name = f"{std_validator} - {yuma_version}"
            if col_name in df.columns:
                columns.append(col_name)
    df = df[columns]

    return df

def generate_ipynb_table(table_data, summary_table):
    custom_css = """
    <style>
        .scrollable-table-container {
            width: 100%; 
            overflow-x: auto;
            overflow-y: hidden;
            white-space: nowrap;
            border: 1px solid #ccc;  
            background-color: hsl(0, 0%, 98%);
        }
        table {
            border-collapse: collapse;
            table-layout: auto;
            width: auto;
        }
        td, th {
            padding: 10px;
            vertical-align: top;
            text-align: center;
        }
    </style>
    """

    # Generate HTML rows
    html_rows = []
    for i in range(len(next(iter(table_data.values())))):  # Number of rows
        row_html = '<tr>'
        for yuma_version in summary_table.columns:
            cell_content = summary_table[yuma_version][i]
            row_html += f'<td>{cell_content}</td>'
        row_html += '</tr>'
        html_rows.append(row_html)

    # Combine rows and create the final table
    html_table = f"""
    <div class="scrollable-table-container">
        <table>
            <thead>
                <tr>{''.join(f'<th>{col}</th>' for col in summary_table.columns)}</tr>
            </thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>
    """
    return custom_css + html_table