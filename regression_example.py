""" Circular regression example: This script runs a simple circular regression analysis
    with simulated data to illustrate how to use the parent class in combination with
    a custom child class and regression variables.

    1. Simulate data
    2. Run regression analysis
    3. Plot the results
"""


if __name__ == '__main__':

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from RegVarsExample import RegVars
    from RegressionChildExample import RegressionChildExample

    # Turn interactive mode on
    plt.ion()

    # ----------------
    # 1. Simulate data
    # ----------------

    # Simulation variables
    n_subj = 10  # number of subjects
    n_trials = 400  # number of trials
    motor_noise = 10  # motor noise
    fixed_LR = 0.3  # fixed learning rate

    # Simulate prediction errors
    delta_t = np.random.normal(0, 50, n_subj * n_trials)

    # Simulate updates corrupted by motor noise
    a_t = np.random.normal(fixed_LR * delta_t, motor_noise)

    # Create data frame for regression
    df_example = pd.DataFrame()
    df_example['delta_t'] = delta_t
    df_example['a_t'] = a_t
    df_example['group'] = 0
    df_example['subj_num'] = np.repeat(np.arange(n_subj), n_trials) + 1

    # --------------------------
    # 2. Run regression analysis
    # --------------------------

    # Define regression variables
    # ---------------------------

    reg_vars = RegVars()
    reg_vars.n_subj = n_subj
    reg_vars.n_ker = 4  # number of kernels for estimation
    reg_vars.n_sp = 5  # number of random starting points
    reg_vars.rand_sp = True  # use random starting points

    # Free parameters
    reg_vars.which_vars = {reg_vars.beta_0: True,  # Intercept
                           reg_vars.beta_1: True,  # prediction error
                           reg_vars.omikron_0: True,  # motor noise
                           reg_vars.omikron_1: False,  # learning rate noise
                           }

    # Select parameters according to selected variables and create data frame
    prior_columns = [reg_vars.beta_0, reg_vars.beta_1, reg_vars.omikron_0, reg_vars.omikron_1]

    # Initialize regression object
    regression = RegressionChildExample(reg_vars)  # regression object instance

    # Translate degrees to radians, which is necessary for our regression model
    df_example['a_t'] = np.deg2rad(df_example['a_t'])
    df_example['delta_t'] = np.deg2rad(df_example['delta_t'])

    # Run regression
    # --------------

    results_df = regression.parallel_estimation(df_example, prior_columns)

    # -------------------
    # 3. Plot the results
    # -------------------

    # Create figure
    f = plt.figure()

    # Create plot grid
    gs_0 = gridspec.GridSpec(1, 1)

    # Create subplot grid
    gs_00 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_0[0])

    # Plot intercept swarm-boxplot
    # ----------------------------

    ax_0 = plt.Subplot(f, gs_00[0, 0])
    f.add_subplot(ax_0)
    sns.boxplot(y="beta_0", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_0, showcaps=False)
    sns.swarmplot(y="beta_0", data=results_df, color='gray', alpha=0.7, size=5, ax=ax_0)

    # Plot learning-rate swarm-boxplot
    # --------------------------------

    ax_1 = plt.Subplot(f, gs_00[0, 1])
    f.add_subplot(ax_1)
    sns.boxplot(y="beta_1", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_1, showcaps=False)
    sns.swarmplot(y="beta_1", data=results_df, color='gray', alpha=0.7, size=5, ax=ax_1)

    # Plot motor-noise swarm-boxplot
    # ------------------------------

    ax_2 = plt.Subplot(f, gs_00[0, 2])
    f.add_subplot(ax_2)
    sns.boxplot(y="omikron_0", data=results_df,
                notch=False, showfliers=False, linewidth=0.8, width=0.15,
                boxprops=dict(alpha=1), ax=ax_2, showcaps=False)
    sns.swarmplot(y="omikron_0", data=results_df, color='gray', alpha=0.7, size=5, ax=ax_2)

    # Delete unnecessary axes
    sns.despine()
    plt.tight_layout()

    # Show plot
    plt.ioff()
    plt.show()
