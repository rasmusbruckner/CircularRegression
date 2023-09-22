""" Parent-Regression Class: Parent class for custom circular regressions """

import numpy as np
import pandas as pd
from scipy.stats import vonmises
from scipy.optimize import minimize
import sys
from time import sleep
from tqdm import tqdm
from itertools import compress
from multiprocessing import Pool


class RegressionParent:
    """ This parent class specifies the instance variables and methods of the common methods of
        circular regression analyses
    """

    def __init__(self, reg_vars):
        """ This function defines the instance variables unique to each instance

            See project-specific RegVars in child class for documentation

        :param reg_vars: Regression-variables-object instance
        """

        # Extract free parameters
        self.which_vars = reg_vars.which_vars

        # Extract fixed parameter values
        self.fixed_coeffs_reg = reg_vars.fixed_coeffs_reg

        # Extract other attributes
        self.n_subj = reg_vars.n_subj
        self.n_ker = reg_vars.n_ker
        self.which_exp = reg_vars.which_exp
        self.show_ind_prog = reg_vars.show_ind_prog
        self.rand_sp = reg_vars.rand_sp
        self.n_sp = reg_vars.n_sp
        self.bnds = reg_vars.bnds

    def parallel_estimation(self, df, prior_columns):
        """ This function manages the parallel estimation of the regression models

        :param df: Data frame containing the data
        :param prior_columns: Selected parameters for repression
        :return: results_df: Data frame containing regression results
        """

        # Inform user about progress
        pbar = None
        if self.show_ind_prog:

            # Inform user
            sleep(0.1)
            print('\nRegression model estimation:')
            sleep(0.1)

            # Initialize progress bar
            pbar = tqdm(total=self.n_subj)

        # Function for progress bar update
        def callback(_):
            """ This callback function updates the progress bar

                futuretodo: put into utilities
            """
            if self.show_ind_prog:
                pbar.update()

        # Initialize pool object for parallel processing
        pool = Pool(processes=self.n_ker)

        # Estimate parameters in parallel
        results = [pool.apply_async(self.estimation,
                                    args=(df[(df['subj_num'] == i + 1)].copy(),),
                                    callback=callback) for i in range(0, self.n_subj)]
        output = [p.get() for p in results]
        pool.close()
        pool.join()

        # Close progress bar
        if self.show_ind_prog and pbar:
            pbar.close()

        # Put all results in data frame
        values = self.which_vars.values()
        columns = list(compress(prior_columns, values))
        columns.append('llh')
        columns.append('group')
        results_df = pd.DataFrame(output, columns=columns)

        return results_df

    def estimation(self, df_subj_input):
        """ This function estimates the coefficients of the mixture model

        :param df_subj_input: Data frame containing subject-specific subset of data
        :return: results_list: List containing regression results
        """

        # Control random number generator for reproducible results
        np.random.seed(123)

        # Get data matrix that is required for the model from child class
        df_subj = self.get_datamat(df_subj_input)

        # Adjust index of this subset of variables
        df_subj.reset_index().rename(columns={'index': 'trial'})

        # Select starting points and boundaries
        # -------------------------------------

        # Extract free parameters
        values = self.which_vars.values()

        # Select boundaries according to free parameters
        bnds = list(compress(self.bnds, values))

        # Initialize with unrealistically high likelihood and no parameter estimate
        min_llh = 100000
        min_x = np.nan

        # Cycle over starting points
        for _ in range(0, self.n_sp):

            # Get project-specific starting points for child class
            x0 = self.get_starting_point()

            # Select starting points according to free parameters
            x0 = np.array(list(compress(x0, values)))

            # Estimate parameters
            res = minimize(self.llh, x0, args=(df_subj,), method='L-BFGS-B', options={'disp': False}, bounds=bnds)

            # Parameter values
            x = res.x

            # Negative log-likelihood
            llh_sum = res.fun

            # Check if cumulated negative log likelihood is lower than the previous
            # one and select the lowest
            if llh_sum < min_llh:
                min_llh = llh_sum
                min_x = x

        # Add results to list
        results_list = list()
        for i in range(len(min_x)):
            results_list.append(float(min_x[i]))

        # Extract group for output
        group = float(list(set(df_subj['group']))[0])

        # Add group and log likelihood to output
        results_list.append(float(min_llh))
        results_list.append(group)

        return results_list

    def llh(self, coeffs, df):
        """ This function computes the likelihood of participant updates, given the specified parameters

        :param coeffs:      Regression coefficients
        :param df:          Data frame containing subset of data
        :return: llh_sum:   Sum of negative log likelihood of mixture model
        """

        # Initialize small value that replaces zero probabilities for numerical stability
        corrected_0_p = 1e-10

        # Extract parameters
        # ------------------

        # Get fixed parameters of regression
        fixed_coeffs = self.fixed_coeffs_reg

        # Initialize coefficient dictionary and counters
        sel_coeffs = dict()  # initialize list with regressor names
        i = 0  # initialize counter

        # futuretodo: maybe as a separate function when used in a different context as well
        # Put selected coefficients in list that is used for the regression
        for key, value in self.which_vars.items():
            if value:
                sel_coeffs[key] = coeffs[i]
                i += 1
            else:
                sel_coeffs[key] = fixed_coeffs[key]

        # Linear regression component
        # ---------------------------

        # Create linear regression matrix
        lr_mat = df[self.update_regressors].to_numpy()

        # Linear regression parameters
        update_regressors = [value for key, value in sel_coeffs.items() if key not in ['omikron_0', 'omikron_1']]

        # Compute predicted update
        a_t_hat = np.sum(lr_mat * update_regressors, 1)

        # Compute standard deviation of update distribution
        up_noise = sel_coeffs['omikron_0'] + sel_coeffs['omikron_1'] * abs(a_t_hat)

        # Convert std of update distribution to radians and kappa
        up_noise_radians = np.deg2rad(up_noise)
        up_var_radians = up_noise_radians ** 2
        kappa_up = 1 / up_var_radians

        # Compute probability density of update
        p_a_t = vonmises.pdf(df['a_t'], loc=a_t_hat, kappa=kappa_up)
        p_a_t[p_a_t == 0] = corrected_0_p  # adjust zeros to small value for numerical stability

        # Check for inf and nan
        if sum(np.isinf(p_a_t)) > 0:
            sys.exit("p_a_t contains infs")
        elif sum(np.isnan(p_a_t)) > 0:
            sys.exit("p_a_t contains nans")

        # Compute log likelihood of linear regression
        llh_reg = np.log(p_a_t)

        # Check for inf and nan
        if sum(np.isinf(llh_reg)) > 0:
            sys.exit("llh_reg contains inf's")
        elif sum(np.isnan(llh_reg)) > 0:
            sys.exit("llh_reg contains nan's")

        # Compute negative log-likelihood
        llh_sum = -1 * (np.nansum(llh_reg))

        return llh_sum

    def get_datamat(self, df_subj_input):
        """ This function raises an error is the get_datamat function is undefined in the
            project-specific regression.
        """
        raise NotImplementedError("Subclass needs to define this.")

    def get_starting_point(self):
        """ This function raises an error is the get_starting-point function is undefined in the
        project-specific regression.
        """
        raise NotImplementedError("Subclass needs to define this.")

