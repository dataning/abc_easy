
"""
ALOG NAV - PRODUCTION-GRADE NAVCASTING
=========================================
Comprehensive implementation with extensive documentation
"""

# Dependencies
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.linalg import sqrtm
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# CONSTANTS
# ============================================================================

MINIMUM_NAV = 1e-6  # Minimum NAV value for numerical stability

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class Estimate:
    """Container for estimation results."""
    nav_estimate: pd.DataFrame  # Columns: 'central', 'lower', 'upper'


@dataclass
class EstimationData:
    """Container for input data."""
    valuations: pd.DataFrame  # GP marks (can have NaNs)
    cashflows: pd.Series  # Cashflows (set to 0 if not using)
    comparables: Optional[pd.DataFrame] = None  # Public comps
    comparables_longer_horizon: Optional[pd.DataFrame] = None  # Extended history
    
    @property
    def dates(self):
        return self.valuations.index
    
    @property
    def number_of_comparables(self):
        return 0 if self.comparables is None else len(self.comparables.columns)
    
    def get_merged_comparables(self):
        """Merge comparables with longer horizon if available."""
        if self.comparables_longer_horizon is None:
            return self.comparables
        return pd.concat([self.comparables_longer_horizon, self.comparables])


@dataclass
class EstimationLogscaleParameters:
    """Container for estimation parameters."""
    valuations_std: pd.Series  # Std dev for each valuation source
    volatility: float  # Private company volatility
    correlations_to_private: pd.Series  # Correlation of each comp to private
    comparables_covariance: Optional[pd.DataFrame] = None  # Comp covariance matrix
    idiosyncratic_growth: float = 0.0  # Growth rate (0 = no growth)


@dataclass
class EstimationInputs:
    """Complete input package for estimation."""
    data: EstimationData
    parameters: EstimationLogscaleParameters


# ############################################################################
# Abstract Estimator Class
# ############################################################################

class AbstractEstimator(ABC):
    """An abstract estimator class, which serves as the basis for the estimation algorithms."""
    
    @abstractmethod
    def __call__(
        self,
        data_input: EstimationInputs,
    ) -> Estimate:
        """Function that estimates the state of the companies using the available valuations and cashflows.
        
        Args:
            data_input (EstimationInputs):
                Data structure containing the data required to perform the estimation
        
        Returns:
            Estimate:
                Dataclass containing the result of the estimation
        """
        pass


# ############################################################################
# Filtering Classes
# ############################################################################

class RollForward(AbstractEstimator):
    """Implements a baseline roll-forward algorithm for private company NAV estimation.
    
    This class provides a simple but reliable estimation method that serves as both a standalone 
    estimator and as an initial estimator for more sophisticated algorithms.
    
    The approach uses the most recent observed valuation as the base estimate, adjusted for 
    subsequent cashflows and growth.
    
    Algorithm Overview:
        1. **Observation Periods**: When valuations are available, compute a weighted average
           based on measurement precision (inverse of standard deviation squared)
        2. **Gap Periods**: When no valuations are observed, roll forward the previous estimate 
           using deterministic growth and cashflow adjustments
        3. **Uncertainty Evolution**: Maintain constant uncertainty during gap periods
    
    Mathematical Framework:
        For periods with observations:
            - **Value Estimate**: nav_t = exp(Sum(w_i * log(obs_nav_i)) / Sum(w_i)) 
              where w_i = 1/sigma_i^2
            - **Uncertainty**: sigma_hat_t = 1/sqrt(Sum(w_i)) (precision-weighted combination)
        
        For periods without observations:
            - **Value Estimate**: nav_t = (nav_(t-1) + cashflow_t) * growth
            - **Uncertainty**: sigma_hat_t = sigma_hat_{t-1} (uncertainty remains constant)
    
    This method is particularly useful for:
        - **Sparse Data Scenarios**: When sophisticated models lack sufficient observations
        - **Baseline Comparisons**: Providing a simple benchmark for complex algorithms
        - **Initial Estimates**: Supplying starting values for iterative optimization methods
        - **Real-time Applications**: When computational efficiency is prioritized
    
    Notes:
        - The method assumes that missing observations represent periods without new information
        - Cashflows are assumed to affect value additively in nominal terms
        - Growth is applied multiplicatively to the base NAV value
        - The first time period must have at least one non-missing observation
        - Uncertainty does not increase during gap periods (conservative assumption)
    """
    
    def __call__(
        self,
        data_input: EstimationInputs,
    ) -> Estimate:
        """Execute the roll-forward estimation algorithm.
        
        This method processes the input data sequentially through time, computing estimates for 
        each period based on available observations or rolling forward previous estimates with 
        adjustments.
        
        The algorithm handles two distinct cases per time period:
            1. **Observation Available**: Compute precision-weighted average of all available valuations
            2. **No Observations**: Roll forward previous estimate with growth and cashflow adjustments
        
        Args:
            data_input (EstimationInputs):
                Complete structured input containing:
                    - **data.valuations**: Private company valuations (may contain NaN values)
                    - **data.cashflows**: Cashflow series indexed by date
                    - **parameters.valuations_std**: Standard deviations for each log(valuation) type
                    - **parameters.idiosyncratic_growth**: logarithm of the growth factor between periods
        
        Returns:
            Estimate:
                Complete estimation results containing:
                    - **nav_estimate**: DataFrame with columns ['central', 'lower', 'upper']
                        * 'central': Optimal NAV estimates indexed by date
                        * 'lower': Lower bound estimates using geometric standard deviations
                        * 'upper': Upper bound estimates using geometric standard deviations
        
        Raises:
            ValueError: If all valuations in the first time period are missing (NaN)
        
        Algorithm Details:
            - **Estimation is performed in the log space**: This is for numerical stability, and 
              because the standard deviation in data_input.parameters.valuations_std is expressed 
              for log quantities.
            
            - **Variable Relationship**:
                - x_t = log(nav_t) represents the log NAV at time t
                - nav_t is the actual NAV in dollars at time t
            
            - **Precision Weighting**: When multiple valuations are available for a single period, 
              they are combined using inverse-variance weighting:
                - Higher precision observations (lower std) receive higher weights
                - Combined uncertainty reflects the precision-weighted information content
            
            - **Roll-Forward Logic**: During gap periods between observations:
                - Previous NAV estimate is scaled by idiosyncratic growth factor
                - Cashflows are added to account for value-affecting distributions/investments
                - Uncertainty remains constant (no additional information or model uncertainty)
            
            - **Edge Cases**:
                - First period must have observations (validation enforced)
                - Missing observations in later periods trigger roll-forward behavior
                - Zero cashflows are handled naturally in the additive adjustment
        
        Example Usage:
            ```python
            estimator = RollForward()
            result = estimator(data_input)
            nav_time_series = result.nav_estimate['central']
            ```
        """
        
        # ====================================================================
        # Data Extraction and Preprocessing
        # ====================================================================
        
        # Extract core data components from structured input
        valuations = data_input.data.valuations  # Private company valuation observations
        cashflows = data_input.data.cashflows  # Cashflow series affecting NAV evolution
        dates = data_input.data.dates  # Time index for the estimation period
        
        # Extract estimation parameters
        log_valuations_std = (
            data_input.parameters.valuations_std
        )  # Measurement uncertainties, expressed in the log space
        idiosyncratic_growth = np.exp(
            data_input.parameters.idiosyncratic_growth
        )  # Growth between periods
        
        # Convert standard deviations to precision weights (inverse variance)
        # Higher precision = lower standard deviation = higher weight in averaging
        inv_log_valuations_var = log_valuations_std.pow(-2)
        
        # ====================================================================
        # Result Storage Initialization
        # ====================================================================
        
        # Initialize result DataFrame with standard structure
        # The estimation is performed in the log space, and later converted to the actual dollar value
        # All estimates start at zero and will be populated sequentially
        nav_estimate = pd.DataFrame(data=0.0, index=dates, columns=["central", "lower", "upper"])
        
        # ====================================================================
        # Critical Validation: First Period Must Have Data
        # ====================================================================
        
        # The roll-forward algorithm requires an initial observation to start from
        # This validation is redundant with EstimationData validation but provides clarity
        if valuations.loc[dates[0]].isna().all():
            raise ValueError("At least one of the values of the first valuation should not be zero")
        
        # ====================================================================
        # Main Estimation Loop
        # ====================================================================
        
        # Process each time period sequentially to build the NAV time series
        for i, date in enumerate(dates):
            # Identify which valuations are observed (not NaN) for this period
            _observed_valuations = valuations.loc[date].notna()
            
            if _observed_valuations.any():
                # ============================================================
                # Case 1: Observations Available
                # ============================================================
                
                # Compute precision-weighted average of available observations
                # This optimally combines multiple valuation sources when available
                
                # Extract observed values and their corresponding precisions
                # Because the stds are expressed for the NAV in the log space, we convert the valuations to log
                log_observed_vals = np.log(valuations.loc[date, _observed_valuations])
                observed_precisions = inv_log_valuations_var[_observed_valuations]
                
                # Precision-weighted average: nav_t = exp(Sum(w_i * log(obs_nav_i)) / Sum(w_i))
                # where w_i = 1/sigma_i^2 is the precision (inverse variance)
                nav_estimate.loc[date, "central"] = np.exp(
                    log_observed_vals @ observed_precisions / observed_precisions.sum()
                )
                
                # Combined uncertainty: 1/sqrt(sum(w_i))
                # The more precise observations we have, the lower our combined uncertainty
                # The geometric_std is used to generate the lower and upper estimates
                geometric_std = np.exp(np.sqrt(1 / observed_precisions.sum()))
            
            else:
                # ============================================================
                # Case 2: No Observations - Roll Forward
                # ============================================================
                
                # Use previous estimate adjusted for growth and cashflows
                # This maintains continuity in the NAV time series during observation gaps
                
                # Roll forward previous NAV with multiplicative growth and additive cashflows
                # Formula: nav_t = (nav_{t-1} + cashflow_t) * growth_factor
                nav_estimate.loc[date, "central"] = (
                    nav_estimate.iloc[i - 1]["central"] + cashflows.loc[date]
                ) * idiosyncratic_growth
                
                # Maintain previous period's uncertainty
                # Conservative assumption: no new information = no change in uncertainty
                geometric_std = geometric_std  # Keep from previous iteration
            
            # Clipping the data to ensure quality
            nav_estimate.loc[date, "central"] = np.clip(
                nav_estimate.loc[date, "central"], a_min=MINIMUM_NAV, a_max=None
            )
            
            # Using the geometric std to create the lower and upper bound estimates
            nav_estimate.loc[date, "lower"] = nav_estimate.loc[date, "central"] / geometric_std
            nav_estimate.loc[date, "upper"] = nav_estimate.loc[date, "central"] * geometric_std
        
        # ====================================================================
        # Result Packaging
        # ====================================================================
        
        # Return results in the standardized Estimate dataclass format
        return Estimate(nav_estimate=nav_estimate)


# ############################################################################
# NAVcasting Class
# ############################################################################

class NAVcasting(AbstractEstimator):
    """Maximum Likelihood Estimation with configurable loss functions for private company NAV estimation.
    
    This class implements a sophisticated state-space model that fuses sparse private company 
    valuations with frequent public comparable observations to produce Bayesian NAV estimates.
    
    The method uses Bayesian Maximum A Posteriori (MAP) estimation with configurable loss functions 
    to balance observation fidelity with smooth temporal dynamics.
    
    Core Innovation:
        Unlike traditional approaches that rely solely on private company data, this estimator 
        leverages the information content from correlated public comparables to:
            1. **Fill observation gaps** during periods without private company valuations
            2. **Reduce estimation uncertainty** through cross-company correlations
            3. **Improve temporal consistency** via joint dynamics modeling
            4. **Provide uncertainty quantification** through posterior covariance estimation
            5. **Configurable robustness** via selectable loss functions for different noise assumptions
    
    Mathematical Framework:
        The estimator implements a Bayesian state-space model:
        
        **State Evolution** (Joint Dynamics in Log Space):
            x_t = x_{t-1} + mu_t + epsilon_t
        
        Here: x_t = log(nav_t) represents the "true log NAV" of the companies
        
        **Observation Model** (Measurement Equations):
            - Private: log(y_t^private) = x_t^private + eta_t^private, 
              eta_t^private ~ specified distribution
            - Comparables: log(y_t^comps) = x_t^comps + eta_t^comps, 
              eta_t^comps ~ specified distribution
        
        **MAP Objective Function**:
            The estimator solves: x* = argmin[-log P(x|data)] where the objective combines:
                - **Observation Cost**: Configurable loss function applied to deviations from observed valuations
                - **Dynamics Cost**: Configurable loss function applied to state transitions using learned covariance structure
    
    Loss Function Options:
        - **Gaussian (L2) Loss** ('obs_potential="gaussian"', 'dyn_potential="gaussian"'):
            - **Mathematical Form**: ||residual||_2^2 = sum(residual^2)
            - **Statistical Assumption**: Gaussian noise model
            - **Characteristics**: Smooth optimization, sensitive to outliers
            - **Best For**: Clean data, analytical tractability, fast computation
        
        - **Laplacian (L1) Loss** ('obs_potential="laplacian"', 'dyn_potential="laplacian"'):
            - **Mathematical Form**: ||residual||_1 = sum|residual|
            - **Statistical Assumption**: Laplacian (double exponential) noise model
            - **Characteristics**: Robust to outliers, promotes sparsity
            - **Best For**: Noisy data, volatile markets, outlier tolerance
    
    Algorithm Variants:
        The class supports two main operating modes:
        
        1. **Fixed Covariance** ('dyn_tune="none"'):
            - Uses pre-specified or empirically estimated covariance matrices
            - Faster computation, suitable when parameters are well-known
            - Single-pass MAP optimization
        
        2. **Adaptive Covariance** ('dyn_tune="EM"'):
            - Iteratively refines covariance matrices using Expectation-Maximization
            - More accurate parameter learning from data
            - Alternates between state estimation (E-step) and parameter updates (M-step)
    
    Key Features:
        - **Configurable Loss Functions**: Choose between Gaussian and Laplacian assumptions
        - **Sparse Observation Handling**: Automatically handles missing private valuations
        - **Multi-Source Integration**: Combines different valuation types with appropriate weights
        - **Cashflow Integration**: Incorporates value-affecting cash distributions/investments
        - **Uncertainty Quantification**: Provides posterior standard deviations for all estimates
        - **Numerical Robustness**: Multiple solver backends and eigenvalue regularization
        - **Scalable Architecture**: Efficient implementation for large time series and many comparables
    
    Attributes:
        dyn_tune: str
            Covariance matrix estimation method ("none" or "EM")
        initial_estimator: AbstractEstimator
            Baseline estimator for initialization and cashflow normalization
        comps_std: float
            Standard deviation assumption for comparable company observations
        EM_num_iters: int (if dyn_tune="EM")
            Maximum iterations for Expectation-Maximization algorithm
        EM_error_thresh: float (if dyn_tune="EM")
            Convergence threshold for EM iterations
        potential_function_dyn: callable
            Loss function for dynamics regularization
        potential_function_obs: callable
            Loss function for observation fidelity
        solver_point_estimate: cp.Problem or None
            Cached CVXPY optimization problem for efficient re-solving
    
    Notes:
        - The optimization is performed in log-space so that the dynamics are affine (it also improves numerical stability)
        - The class automatically handles missing observations and irregular time series
        - Covariance matrices are regularized to ensure positive definiteness
        - Multiple solver backends provide robustness against numerical issues
        - The estimator maintains computational efficiency through problem structure caching
        - Loss function selection affects optimization characteristics and robustness properties
    """
    
    def __init__(
        self,
        dyn_tune: str = "none",
        initial_estimator: AbstractEstimator | None = None,
        comps_std: float = 0.01,
        obs_potential: str = "laplacian",
        dyn_potential: str = "gaussian",
        kw_args: dict[str, float] | None = None,
    ) -> None:
        """Initialize the NAVcasting estimator with specified configuration.
        
        This constructor sets up the estimator's operating parameters and initializes the 
        optimization infrastructure. The configuration determines the balance between computational 
        efficiency, estimation accuracy, and robustness to outliers.
        
        Args:
            dyn_tune: str, default="none"
                Method for covariance matrix estimation:
                    - **"none"**: Use fixed covariance matrices (faster, less adaptive)
                        * Suitable when volatilities and correlations are well-known
                        * Single optimization pass per estimation
                        * Recommended for real-time applications
                    - **"EM"**: Expectation-Maximization for adaptive covariance learning
                        * Iteratively learns optimal parameters from data
                        * Higher computational cost but better accuracy
                        * Recommended when parameters are uncertain and there is a lot of data available
            
            initial_estimator: AbstractEstimator, optional
                Baseline estimator used for:
                    - **Initialization**: Starting values for optimization
                    - **Cashflow Normalization**: Reference values for log-space adjustments
                    - **Single-Period Fallback**: Direct result when T=1
                If None, defaults to RollForward() which provides robust baseline estimates.
            
            comps_std: float, default=0.01
                Standard deviation assumption for comparable company market cap observations.
                This parameter controls the relative weight of comparable vs. private company data:
                    - **Lower values** (e.g., 0.005): Trust comparables more, tighter coupling
                    - **Higher values** (e.g., 0.02): Trust comparables less, looser coupling
                    - **Typical range**: 0.005 - 0.02 for daily market data
            
            obs_potential: str, default="laplacian"
                Loss function type for observation fidelity term in MAP objective:
                    - **"gaussian"**: L2 (squared) loss function
                        * Mathematical form: ||residual||_2^2 = sum(residual^2)
                        * Characteristics: Smooth optimization, sensitive to outliers
                        * Best for: Clean data, fast computation, analytical tractability
                    - **"laplacian"**: L1 (absolute) loss function
                        * Mathematical form: ||residual||_1 = sum|residual|
                        * Characteristics: Robust to outliers, promotes sparsity
                        * Best for: Noisy data, volatile markets, outlier tolerance
            
            dyn_potential: str, default="gaussian"
                Loss function type for dynamics regularization term in MAP objective:
                    - **"gaussian"**: L2 (squared) loss function
                        * Encourages smooth state transitions
                        * Penalizes large deviations quadratically
                        * Standard choice for continuous dynamics
                    - **"laplacian"**: L1 (absolute) loss function
                        * Promotes sparse transitions (occasional jumps)
                        * Linear penalty for large deviations
                        * Suitable for regimes with infrequent value adjustments
            
            kw_args: dict[str, float], optional
                Additional configuration parameters for advanced users:
                
                **For dyn_tune="EM"** (Expectation-Maximization parameters):
                    - **EM_num_iters** (int, default=100): Maximum iterations for EM algorithm
                        * Higher values allow more thorough convergence but increase computation time
                        * Typical range: 20-1000 depending on data complexity
                    - **EM_error_thresh** (float, default=1e-5): Convergence tolerance
                        * Smaller values require tighter convergence but may increase iterations
                        * Measured as relative change in covariance matrix infinity norm
                
                **Example usage**:
                    ```python
                    # High-accuracy configuration with tight convergence
                    kw_args = {"EM_num_iters": 1000, "EM_error_thresh": 1e-6}
                    
                    # Fast configuration for real-time applications
                    kw_args = {"EM_num_iters": 20, "EM_error_thresh": 1e-4}
                    ```
        
        Raises:
            ValueError: If dyn_tune is not "none" or "EM"
            ValueError: If obs_potential is not "gaussian" or "laplacian"
            ValueError: If dyn_potential is not "gaussian" or "laplacian"
        """
        
        # ====================================================================
        # Input Validation and Defaults
        # ====================================================================
        
        # Validate potential function types
        valid_potentials = {"gaussian", "laplacian"}
        if obs_potential not in valid_potentials:
            raise ValueError(
                f'obs_potential must be one of {valid_potentials}, got "{obs_potential}"'
            )
        
        if dyn_potential not in valid_potentials:
            raise ValueError(
                f'dyn_potential must be one of {valid_potentials}, got "{dyn_potential}"'
            )
        
        # Validate dynamics tuning method
        if dyn_tune not in ("none", "EM"):
            raise ValueError(f'dyn_tune must be "none" or "EM", got "{dyn_tune}"')
        
        # Set default initial estimator if none provided
        # RollForward provides robust baseline estimates for initialization
        if initial_estimator is None:
            initial_estimator = RollForward()
        
        # Initialize keyword arguments dictionary for additional parameters
        if kw_args is None:
            kw_args = {}
        
        # ====================================================================
        # Core Configuration Storage
        # ====================================================================
        
        # Store primary configuration parameters
        self.method_dyn_tune = dyn_tune  # Covariance estimation method
        self.initial_estimator = initial_estimator  # Baseline estimator for initialization
        self.comps_std = comps_std  # Standard deviation for comparable observations
        
        # ====================================================================
        # Algorithm-Specific Parameter Configuration
        # ====================================================================
        
        # Configure Expectation-Maximization parameters (if applicable)
        if dyn_tune == "EM":
            # Set default EM algorithm parameters
            # These values balance convergence quality with computational efficiency
            self.EM_num_iters = 100  # Maximum iterations for convergence
            self.EM_error_thresh = 1e-5  # Relative change threshold for termination
            
            # Override defaults with user-specified parameters
            # This allows fine-tuning of algorithm behavior for specific applications
            for key, value in kw_args.items():
                setattr(self, key, value)
        
        # ====================================================================
        # Optimization Infrastructure Initialization
        # ====================================================================
        
        # Initialize solver cache as None - will be created on first use
        # This Lazy Loading approach avoids unnecessary memory allocation
        self.solver_point_estimate: cp.Problem | None = None
        
        # ====================================================================
        # Loss Function Configuration
        # ====================================================================
        
        # Configure the potential functions for the MAP objective
        # These functions define the shape of the penalty terms in the optimization
        
        # Dynamics potential function: penalizes large state transitions
        if dyn_potential == "gaussian":
            # cp.sum_squares creates quadratic penalties consistent with Gaussian assumptions
            # This encourages smooth evolution of NAV values over time
            self.potential_function_dyn = cp.sum_squares
        elif dyn_potential == "laplacian":
            # L1 penalty promotes sparse transitions (occasional jumps)
            # The sqrt(2) factor ensures equivalent scale to Gaussian case under normal conditions
            self.potential_function_dyn = lambda x: cp.sum(cp.abs(x * np.sqrt(2)))
        
        # Observation potential function: penalizes deviations from observed values
        if obs_potential == "gaussian":
            # cp.sum_squares creates quadratic penalties for measurement errors
            # This ensures the solution remains close to actual observations
            self.potential_function_obs = cp.sum_squares
        elif obs_potential == "laplacian":
            # L1 penalty provides robustness against outlying observations
            # The sqrt(2) factor ensures equivalent scale to Gaussian case under normal conditions
            self.potential_function_obs = lambda x: cp.sum(cp.abs(x * np.sqrt(2)))
        
        # Select the solver order once so the solve loop can prioritise conic solvers when needed
        if obs_potential == "gaussian" and dyn_potential == "gaussian":
            self._solver_sequence = ["OSQP", "CLARABEL", "SCS"]
        else:
            # Laplacian losses introduce L1 terms which are better handled by conic solvers like CLARABEL
            self._solver_sequence = ["CLARABEL", "SCS", "OSQP"]
    
    def _compute_hessian_of_objective(
        self,
        data_input: EstimationInputs,
        cov_private_and_comps: np.ndarray,
    ) -> np.ndarray:
        """Computes the Hessian matrix of the negative log-posterior (MAP) objective function.
        
        This method constructs the second derivative matrix of the optimization objective, which 
        consists of two main components:
            1. Observation term: penalties for deviations from observed valuations
            2. Dynamics term: penalties for non-smooth state evolution between time periods
        
        The resulting Hessian is used for uncertainty quantification and can provide the covariance 
        matrix of the estimated values via its inverse.
        
        Args:
            data_input (EstimationInputs):
                Structured input containing:
                    - Valuation data for the private company with measurement uncertainties
                    - Comparable companies market cap data
                    - Parameters including standard deviations and correlations
            
            cov_private_and_comps (np.ndarray):
                Joint covariance matrix of shape (1+num_comparables, 1+num_comparables) governing 
                the dynamics between private company and comparables:
                    - [0,0]: Private company variance
                    - [0,1:] and [1:,0]: Cross-covariances with comparables
                    - [1:,1:]: Covariance matrix among comparables
        
        Returns:
            np.ndarray:
                Hessian matrix of shape (T*n, T*n) where T is the number of time periods and n is 
                the number of companies (1 private + num_comparables).
                
                Matrix structure (block form):
                    - Diagonal blocks: Observation precision + dynamics contributions
                    - Off-diagonal blocks: Cross-time dynamics coupling terms
                
                The matrix represents d^2(-log p(x|data)) where x are the log NAV states.
        
        Example structure for T=4 time periods (4x4 blocks):
        
                    t=1         t=2         t=3         t=4
            t=1  [Σ^-1+obs_1  -Σ^-1       0           0      ]
            t=2  [-Σ^-1       2Σ^-1+obs_2 -Σ^-1       0      ]
            t=3  [0          -Σ^-1        2Σ^-1+obs_3 -Σ^-1  ]
            t=4  [0           0          -Σ^-1        Σ^-1+obs_4]
        
        Where:
            - Σ^-1: Inverse covariance matrix from dynamics (nxn block)
            - obs_t: Time-specific diagonal matrix of observation precisions (nxn block)
                     Only takes into account companies/valuations observed at time t (excludes NaNs)
            - Each entry represents an (nxn) block for n companies
        
        Notes:
            - The Hessian is constructed in blocks corresponding to time periods
            - Observation terms contribute only to diagonal elements (independent measurements)
            - Dynamics terms create a tridiagonal block structure reflecting temporal coupling
            - The matrix is guaranteed to be positive definite for well-posed problems
        """
        
        # ====================================================================
        # Problem Dimensions and Setup
        # ====================================================================
        
        number_of_dates = len(data_input.data.dates)
        num_priva_and_comps = 1 + data_input.data.number_of_comparables
        
        # Convert valuation standard deviations to precision (inverse variance)
        # Higher precision = lower uncertainty = higher weight in objective
        inv_var = data_input.parameters.valuations_std.pow(-2)
        
        # Initialize the Hessian matrix with zeros
        # Each company-time pair gets one row/column in the final matrix
        hessian = np.zeros(
            (num_priva_and_comps * number_of_dates, num_priva_and_comps * number_of_dates)
        )
        
        # ====================================================================
        # Observation Component of the Hessian
        # ====================================================================
        
        # The observation terms contribute only to the diagonal of the Hessian
        # since measurement errors are assumed independent across time and companies
        
        # For private company valuations: weight by measurement precision
        # Only dates with actual observations contribute to the objective
        inv_var_valuations_part = data_input.data.valuations.notna() @ inv_var
        inv_var_valuations_part.name = "Inverse of volatility, valuations part"
        
        if data_input.data.comparables is not None:
            # For comparable companies: use uniform precision (self.comps_std)
            # Each non-missing observation contributes 1/σ^2 to the diagonal
            inv_var_comparables_part = data_input.data.comparables.notna() / (self.comps_std**2)
            
            # Combine precisions for all companies and time periods
            # This creates a time series of precision vectors
            merged_obs = pd.merge(
                left=inv_var_valuations_part,
                left_index=True,
                right=inv_var_comparables_part,
                right_index=True,
            )
        else:
            merged_obs = inv_var_valuations_part
        
        # Fill the main diagonal with observation precisions
        # The diagonal structure reflects independence of observations
        np.fill_diagonal(hessian, merged_obs.to_numpy().flatten())
        
        # ====================================================================
        # Dynamics Component of the Hessian
        # ====================================================================
        
        # The dynamics terms enforce smoothness between consecutive time periods
        # They create a block-tridiagonal structure in the Hessian matrix
        
        # Compute the precision matrix for the dynamics (inverse of covariance)
        # Symmetrize the covariance matrix to handle numerical precision issues
        inv_cov_private_and_comps = np.linalg.inv(
            0.5 * (cov_private_and_comps + cov_private_and_comps.T)
        )
        
        # Construct the block-tridiagonal dynamics structure:
        # For a 4-period, single company example, the pattern is:
        #   [+2Σ^-1    -Σ^-1      0         0     ]  ← Current period coupling
        #   [-Σ^-1     +2Σ^-1    -Σ^-1      0     ]  ← Interior periods (coupled both ways)
        #   [0        -Σ^-1      +2Σ^-1    -Σ^-1  ]  ← Interior periods (coupled both ways)
        #   [0         0         -Σ^-1     +Σ^-1  ]  ← Final period coupling
        #
        # Using Kronecker products:
        # The dynamics part of the Hessian can be constructed efficiently as:
        #   H_dynamics = kron(structure, Σ^-1) where structure is the tridiagonal pattern:
        #   structure = [[1,   -1,  0,  0],     # First row: [1, -1, 0, ...]
        #                [-1,   2, -1,  0],     # Interior: [-1, 2, -1, ...]
        #                [0,   -1,  2, -1],     # Interior: [0, -1, 2, -1, ...]
        #                [0,    0, -1,  1]]     # Last row: [..., 0, -1, 1]
        
        structure = (
            2 * np.diag(np.ones(number_of_dates))
            - np.diag(np.ones(number_of_dates - 1), 1)
            - np.diag(np.ones(number_of_dates - 1), -1)
        )
        structure[0, 0] = 1
        structure[-1, -1] = 1
        
        hessian += np.kron(structure, inv_cov_private_and_comps)
        
        return hessian
    
    def _point_estimate(
        self,
        data_input: EstimationInputs,
        cov_private_and_comps: np.ndarray,
        initial_private_estimate: pd.Series,
    ) -> tuple[pd.Series, np.ndarray]:
        """Computes point estimates for private company NAV using Maximum Likelihood Estimation.
        
        This method solves a convex optimization problem to estimate the Net Asset Value (NAV) of a 
        private company by leveraging:
            1. Sparse valuation observations from the private company
            2. Continuous market data from comparable public companies
            3. Cashflow information that affects the private company's value
            4. Prior covariance structure between the private company and comparables
        
        The estimation process uses a state-space model where:
            - States represent log NAV values for the private company and comparables
            - Observations are noisy measurements of these underlying states
            - Dynamics follow a correlated random walk with drift (for cashflows/growth)
        
        Args:
            data_input (EstimationInputs):
                Structured input containing:
                    - Valuation data for the private company (sparse, with measurement uncertainty)
                    - Market cap data for comparable public companies (frequent observations)
                    - Cashflow series affecting private company value
                    - Parameter specifications (volatilities, correlations, growth)
            
            cov_private_and_comps (np.ndarray):
                Covariance matrix of shape (1+num_comparables, 1+num_comparables) governing the 
                joint dynamics between the private company and its public comparables.
                    - [0,0]: Private company variance
                    - [0,1:] and [1:,0]: Cross-covariances with comparables
                    - [1:,1:]: Covariance matrix among comparable companies
            
            initial_private_estimate (pd.Series):
                Initial NAV estimates for the private company indexed by date.
                Used to normalize cashflows in the log-space dynamics model.
                Shape: (num_time_periods,)
        
        Returns:
            tuple[pd.Series, np.ndarray]:
                - point_estimates (pd.Series): Estimated NAV values for the private company 
                  indexed by date, in original dollar units (not log-transformed)
                - disturbance_estimates (np.ndarray): Estimated dynamic disturbances of shape 
                  (num_companies, num_time_periods-1) representing changes in log values between 
                  consecutive time periods not explained by the dynamics
        
        Raises:
            ValueError: If initial_private_estimate or cov_private_and_comps contain NaNs or Infs
            SolverError: If the convex optimization problem fails to converge
            ValueError: If the outputs are either NaN or Inf
        
        Notes:
            - The optimization is performed in log-space so that the dynamics are affine (it also 
              improves numerical stability)
            - Cashflows are normalized by initial estimates to handle scale differences
            - Missing observations are automatically handled through the sparse structure
            - The solver tries multiple backends (OSQP, CLARABEL, SCS) for robustness
        """
        
        # Extract cashflow data for the private company
        # This will be used to adjust the dynamics equation
        cashflows = data_input.data.cashflows
        idiosyncratic_growth = np.exp(data_input.parameters.idiosyncratic_growth)
        
        # Check there are no NaNs or Inf in the initial_private_estimate and cov_private_and_comps
        if not np.isfinite(initial_private_estimate).all():
            raise ValueError("initial_private_estimate contains either NaNs or Infs")
        
        if not np.isfinite(cov_private_and_comps).all():
            raise ValueError("cov_private_and_comps contains either NaNs or Infs")
        
        # ====================================================================
        # Solver Initialization (Lazy Loading)
        # ====================================================================
        
        # Create the optimization problem structure only once for efficiency
        # The solver is cached to avoid reconstruction overhead on repeated calls
        if self.solver_point_estimate is None:
            self.solver_point_estimate = self._create_solver_point_estimate(data_input=data_input)
        
        # ====================================================================
        # Parameter Updates for Current Estimation
        # ====================================================================
        
        # Update the precision matrix (inverse covariance) for dynamics regularization
        # We use the matrix square root for numerical stability in the quadratic form
        # sqrt(inv(Σ))^T @ x creates the term x^T @ inv(Σ) @ x in the objective
        
        # Symmetrize the covariance matrix to avoid numerical issues
        cov_symmetric = 0.5 * (cov_private_and_comps + cov_private_and_comps.T)
        
        # Use eigendecomposition for robust square root computation
        # This ensures we get real values even with numerical precision issues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_symmetric)
        
        # Clip small negative eigenvalues to zero (numerical precision issues)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        
        # Compute sqrt(inv(Σ)) via eigendecomposition: inv(Σ) = V @ diag(1/λ) @ V.T
        # sqrt(inv(Σ)) = V @ diag(1/sqrt(λ)) @ V.T
        precision_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        
        # Ensure result is real (remove any tiny imaginary parts from numerical errors)
        precision_sqrt = np.real(precision_sqrt)
        
        self.solver_point_estimate.param_dict["half_cov_private_and_comps"].value = precision_sqrt
        
        # Compute normalized cashflows for the dynamics equation
        # In log-space: log(nav[t]) = log((nav[t-1] + cashflow[t]) * growth)
        # Which is equivalent to log(nav[t-1]) + log((1 + cashflow[t]/nav[t-1]) * growth) as long as nav[t-1] > 0
        # The part log((1 + cashflow[t]/nav[t-1]) * growth) is treated as a constant representing the expected change,
        # which allows us to have linear dynamics, i.e., a convex problem
        exp_private_expected_change = (
            1 + cashflows[1:].to_numpy() / initial_private_estimate[:-1].to_numpy()
        ) * idiosyncratic_growth
        
        # Clip to ensure log(adjustment) is well conditioned
        # This maintains numerical stability and prevents unrealistic value destruction
        # This is the same as enforcing that initial_private_estimate + cashflows >= MINIMUM_NAV
        exp_private_expected_change_clipped = np.clip(
            exp_private_expected_change,
            a_min=MINIMUM_NAV / initial_private_estimate[:-1].to_numpy(),
            a_max=None,
        )
        
        # Convert to log-space and pass it to the optimizer
        private_expected_change_clipped = np.log(exp_private_expected_change_clipped)
        self.solver_point_estimate.param_dict[
            "private_expected_change"
        ].value = private_expected_change_clipped
        
        # ====================================================================
        # Solve the Optimization Problem
        # ====================================================================
        
        # Attempt solving with multiple solvers for robustness
        # Solver sequence prioritises conic solvers when the objective includes L1 terms
        status = "not optimal"
        solve_history: list[str] = []
        
        for solver_name in self._solver_sequence:
            self.solver_point_estimate.solve(solver=solver_name)
            status = self.solver_point_estimate.status
            solve_history.append(f"{solver_name}: {status}")
            if status == "optimal":
                break
        
        if status != "optimal":
            history = " -> ".join(solve_history)
            raise cp.SolverError(f"Could not solve the optimization. Attempts: {history}")
        
        # The 8 lines above could be substituted by this, the problem is that this prints a message 
        # everytime that the solver solves the problem, and that's annoying
        # self.solver_point_estimate.solve(
        #     solver_path=["OSQP", "CLARABEL", "SCS"],
        # )
        
        # ====================================================================
        # Extract and Transform Results
        # ====================================================================
        
        # Extract log estimates for the private company (first row, excluding first time period)
        # The solver estimates states for all time periods, but we want the refined estimates
        log_private_estimates = self.solver_point_estimate.var_dict["log_estim_all"].value[0]
        
        # Transform back to original dollar units from log-space
        point_estimates = pd.Series(
            np.exp(log_private_estimates),
            index=data_input.data.dates,
            name='central',
        )
        
        # Make sure that we do not have any NANs or Inf
        if not np.isfinite(point_estimates).all():
            raise ValueError("point_estimate results contains either NaNs or Infs")
        
        # Extract dynamic disturbances for all companies
        # Shape: (num_companies, num_time_periods-1)
        # These represent the unexplained innovations in the state evolution
        disturbance_estimates = self.solver_point_estimate.var_dict["d_dyn"].value.T
        
        # Make sure that we do not have any NANS or Inf
        if not np.isfinite(disturbance_estimates).all():
            raise ValueError("disturbance_estimate contains either NaNs or Infs")
        
        return point_estimates, disturbance_estimates
    
    def _create_solver_point_estimate(self, data_input: EstimationInputs) -> cp.Problem:
        """Creates and returns a convex optimization problem for point estimation in a dynamic system.
        
        This method constructs a convex optimization problem that estimates the log NAV of a private 
        company and comparable companies over time. The optimization balances two objectives:
            1. Minimizing prediction errors for observed valuations (observation cost)
            2. Ensuring smooth dynamics by penalizing large changes in the state variables (dynamics cost)
        
        The optimization variables represent:
            - log_estim_all: Log estimates for all companies (private + comparables) across all time periods
            - d_dyn: Dynamic disturbances that capture unexpected changes in company values
        
        Args:
            data_input (EstimationInputs):
                Structured input containing:
                    - Valuation data for the private company (sparse, with measurement uncertainty)
                    - Market cap data for comparable public companies (frequent observations)
                    - Cashflow series affecting private company value
                    - Parameter specifications (volatilities, correlations, growth rates)
        
        Returns:
            cp.Problem: A CVXPY optimization problem ready to be solved. The problem is formulated 
                       as a convex quadratic program (or other convex form depending on potential 
                       functions) that can be efficiently solved to obtain point estimates.
        
        Raises:
            ValueError: If the resulting problem is not DPP (Disciplined Parametrized Programming) compliant.
        
        Notes:
            - The method assumes log-normal dynamics for company values
            - The first row of log_estim_all corresponds to the private company
            - Subsequent rows correspond to comparable companies (if provided). The order is the same 
              as for input_data.data.comparables
            - The dynamics constraint enforces a random walk with drift for all companies
            - Missing observations are automatically handled by skipping them in the cost function
        """
        
        # ====================================================================
        # Prepare the data
        # ====================================================================
        
        log_valuations: pd.DataFrame = np.log(data_input.data.valuations)
        
        if data_input.data.comparables is not None:
            log_comps: pd.DataFrame = np.log(data_input.data.comparables)
        else:
            log_comps = None
        
        valuations_std = data_input.parameters.valuations_std
        
        # Convert standard deviations to inverse weights for optimization
        # Higher uncertainty (larger std) leads to lower weight in the objective function
        inv_priva_std_series = 1 / valuations_std
        inv_comps_std = 1 / self.comps_std
        
        # Convert to numpy for efficient computation in CVXPY
        inv_priva_std = inv_priva_std_series.to_numpy()
        
        # ====================================================================
        # Problem Dimension setup
        # ====================================================================
        
        number_of_dates = len(data_input.data.dates)
        num_priva_and_comps = 1 + data_input.data.number_of_comparables
        
        # ====================================================================
        # Optimization Variables
        # ====================================================================
        
        # log_estim_all[i, t] = log estimate for company i at time t
        # Row 0: private company, Rows 1+: comparable companies
        log_estim_all = cp.Variable(
            shape=(num_priva_and_comps, number_of_dates), name="log_estim_all"
        )
        
        # d_dyn[i, t] = dynamic disturbance for company i between time t and t+1
        # These capture unexpected changes not explained by the deterministic drift
        d_dyn = cp.Variable(shape=(num_priva_and_comps, number_of_dates - 1), name="d_dyn")
        
        # ====================================================================
        # Problem Parameters (to be set later)
        # ====================================================================
        
        # Normalized cashflows for the private company (affects log dynamics)
        private_expected_change = cp.Parameter(
            shape=(number_of_dates - 1), name="private_expected_change"
        )
        
        # Square root of inverse covariance matrix for regularizing dynamics
        half_cov_private_and_comps = cp.Parameter(
            shape=(num_priva_and_comps, num_priva_and_comps), name="half_cov_private_and_comps"
        )
        
        # ====================================================================
        # Dynamics Constraints
        # ====================================================================
        
        # Enforce state evolution equations:
        # For private company: log_val[t] = log_val[t-1] + expected_change[t] (from cashflow and growth) + disturbance[t]
        # For comparables: log_val[t] = log_val[t-1] + disturbance[t] (no cashflows or drift)
        
        dynamics_constraint = [
            # Private company dynamics (row 0)
            # Change in log value = observed change - expected change (cashflow + growth)
            d_dyn[0, :] == (log_estim_all[0, 1:] - log_estim_all[0, :-1] - private_expected_change),
        ]
        
        if log_comps is not None:
            # Comparable companies dynamics (rows 1+)
            # Simple random walk: change in log value = disturbance
            dynamics_constraint.append(
                d_dyn[1:, :] == log_estim_all[1:, 1:] - log_estim_all[1:, :-1]
            )
        
        # ====================================================================
        # Cost Function Construction
        # ====================================================================
        
        # The potential functions are defined in the class constructor and allow
        # different loss functions (quadratic for Gaussian, L1 for Laplace, etc.)
        
        # Dynamics cost: penalize large disturbances using the covariance structure
        # This encourages smooth evolution consistent with the estimated correlations
        cost_dyn = self.potential_function_dyn(half_cov_private_and_comps @ d_dyn)
        
        # Observation cost: penalize deviations from observed valuations
        obs_terms_t = []
        
        # Process each time period separately to handle missing observations
        for i, date in enumerate(log_valuations.index):
            # Private company valuations
            _observed_valuations = log_valuations.loc[date].notna()
            
            if _observed_valuations.any():
                # Weight prediction errors by inverse standard deviations
                # More reliable observations (lower std) get higher weight
                obs_terms_t.append(
                    self.potential_function_obs(
                        cp.multiply(
                            log_estim_all[0, i]
                            - log_valuations.loc[date, _observed_valuations].to_numpy(),
                            inv_priva_std[_observed_valuations],
                        )
                    )
                )
            
            # Comparable companies observations
            if log_comps is not None:
                _observed_comps = log_comps.loc[date].notna()
                
                if _observed_comps.any():
                    # Apply uniform weighting for comparable companies
                    # Selecting only the comparables that have observations
                    obs_terms_t.append(
                        self.potential_function_obs(
                            (
                                log_estim_all[1:, i][_observed_comps]
                                - log_comps.loc[date, _observed_comps]
                            )
                            * inv_comps_std
                        )
                    )
        
        cost_obs = cp.sum(obs_terms_t)
        
        # Total objective: balance observation fidelity with smooth dynamics
        cost = cost_dyn + cost_obs
        
        # ====================================================================
        # Problem Formulation
        # ====================================================================
        
        prob = cp.Problem(
            cp.Minimize(cost),
            constraints=[*dynamics_constraint, log_estim_all >= np.log(MINIMUM_NAV)],
        )
        
        # Verify the problem follows Disciplined Parametrized Programming rules
        # This ensures it can be efficiently solved and parameters can be updated
        if not prob.is_dpp():
            raise ValueError("Problem is not DPP when it should be")
        
        return prob
    
    def _compute_cov_private_and_comps(
        self, data_input: EstimationInputs, effective_zero: float = 1e-8
    ) -> np.ndarray:
        """Constructs the joint covariance matrix between the private company and its public comparables.
        
        This method builds a covariance matrix that captures the joint dynamics between:
            1. The private company (using user-specified volatility)
            2. Public comparable companies (using empirical or provided covariance)
            3. Cross-correlations between the private company and each comparable
        
        The resulting matrix is used in the optimization to regularize the joint evolution of all 
        companies, ensuring that the private company's estimated dynamics are consistent with the 
        behavior of its public comparables.
        
        Args:
            data_input (EstimationInputs):
                Structured input containing all data and parameters needed for estimation:
                    - Private company volatility (parameters.volatility)
                    - Comparable companies' market data (data.comparables)
                    - Cross-correlations (parameters.correlations_to_private)
                    - Optional pre-computed comparable covariance (parameters.comparables_covariance)
            
            effective_zero (float, optional):
                Threshold for eigenvalue regularization. Eigenvalues below this threshold are 
                replaced to ensure positive definiteness. Defaults to 1e-8.
        
        Returns:
            np.ndarray:
                Joint covariance matrix of shape (1+num_comparables, 1+num_comparables) where:
                    - [0,0]: Private company variance (volatility^2)
                    - [0,1:]: Covariances between private company and each comparable
                    - [1:,0]: Symmetric covariances (same as [0,1:])
                    - [1:,1:]: Covariance matrix among comparable companies
                
                The matrix is guaranteed to be positive definite through eigenvalue regularization.
        
        Raises:
            ValueError: If correlations and comparables data are inconsistent
            LinAlgError: If eigendecomposition fails (rare)
        
        Notes:
            - If no comparables are provided, returns a 1x1 matrix with private company variance
            - Comparable covariances are estimated from log-returns if not provided
            - The method ensures positive definiteness through eigenvalue clipping
            - Cross-covariances use the formula: cov(i,j) = corr(i,j) * std(i) * std(j)
        """
        
        # Extract the private company's volatility (standard deviation)
        # This represents the uncertainty in the private company's log-value changes
        private_std = data_input.parameters.volatility
        
        # ====================================================================
        # Handle Simple Case: No Comparables
        # ====================================================================
        
        # If there are no comparable companies, return a 1x1 covariance matrix
        # containing only the private company's variance
        if data_input.data.comparables is None:
            return np.array([[private_std**2]])
        
        # ====================================================================
        # Compute Comparable Companies Covariance
        # ====================================================================
        
        # Option 1: Use provided covariance matrix (if available)
        if data_input.parameters.comparables_covariance is not None:
            # User has provided a pre-computed covariance matrix for comparables
            # Make a copy to avoid modifying the original data
            cov_comps = data_input.parameters.comparables_covariance.to_numpy(copy=True)
        else:
            # Option 2: Estimate covariance from market data
            # Compute empirical covariance from log-returns of comparable companies
            # This uses both regular and extended historical data if available
            merged_comparables = data_input.data.get_merged_comparables()
            log_prices = np.log(merged_comparables)
            
            # Calculate daily returns (first differences of log prices)
            log_returns = log_prices.diff()
            
            # Remove periods where all companies have missing data
            log_returns_clean = log_returns.dropna(how="all")
            
            # Compute sample covariance matrix and convert to numpy array
            cov_comps = log_returns_clean.cov().to_numpy()
        
        # ====================================================================
        # Extract Cross-Correlations
        # ====================================================================
        
        # Get correlations between private company and each comparable
        corr_to_private = data_input.parameters.correlations_to_private.to_numpy()
        
        # ====================================================================
        # Compute Cross-Covariances
        # ====================================================================
        
        # Convert correlations to covariances using the formula:
        # Cov(X,Y) = Corr(X,Y) * Std(X) * Std(Y)
        # Here: Cov(private, comp_i) = Corr(private, comp_i) * private_std * comp_i_std
        # where comp_i_std = sqrt(diagonal element of comparable covariance matrix)
        comparable_stds = np.sqrt(np.diag(cov_comps))
        cov_to_private = corr_to_private * private_std * comparable_stds
        
        # ====================================================================
        # Construct Joint Covariance Matrix
        # ====================================================================
        
        # Build the full covariance matrix using block matrix construction:
        # Structure:
        #   [[private_var      cov_to_private (row vector)],
        #    [cov_to_private   cov_comps (matrix)         ]]
        
        private_variance = private_std**2
        prelim_cov_private_and_comps = np.block(
            [[private_variance, cov_to_private], [cov_to_private[:, None], cov_comps]]
        )
        
        # ====================================================================
        # Ensure Positive Definiteness
        # ====================================================================
        
        # The constructed matrix might not be positive definite due to:
        # 1. Inconsistent correlations (e.g., circular correlation constraints)
        # 2. Numerical precision issues
        # 3. Small sample sizes in empirical covariance estimation
        #
        # We fix this by eigenvalue regularization: replace small/negative eigenvalues
        
        # Compute eigendecomposition: A = Q @ diag(λ) @ Q.T
        eigenvalues, eigenvectors = np.linalg.eig(prelim_cov_private_and_comps)
        
        # Identify problematic eigenvalues (too small or negative)
        problematic_mask = eigenvalues <= effective_zero
        
        if problematic_mask.any():
            # Replace problematic eigenvalues with a small positive value
            # Use a fraction of the median of the "good" eigenvalues to maintain scale
            good_eigenvalues = eigenvalues[~problematic_mask]
            
            if len(good_eigenvalues) > 0:
                replacement_value = 1e-3 * np.median(good_eigenvalues)
            else:
                # Fallback: if all eigenvalues are problematic, use a small default
                replacement_value = 1e-6
            
            eigenvalues[problematic_mask] = replacement_value
            
            # Reconstruct the positive definite matrix
            # A_PD = Q @ diag(λ_regularized) @ Q.T
            regularized_cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            return regularized_cov_matrix
        
        return prelim_cov_private_and_comps
    
    def _tune_dynamics_covar_with_EM(
        self,
        data_input: EstimationInputs,
        cov_private_and_comps: np.ndarray,
        initial_private_estimate: pd.Series,
    ) -> tuple[pd.Series, np.ndarray]:
        """Estimates the dynamics covariance matrix using the Expectation Maximization (EM) algorithm.
        
        This method implements an iterative EM algorithm to simultaneously estimate:
            1. The optimal NAV time series for the private company
            2. The dynamics covariance matrix governing joint evolution of private and comparable companies
        
        The algorithm alternates between:
            - E-step: Given current covariance estimate, compute optimal state sequence (MAP estimation)
            - M-step: Given current state sequence, update covariance matrix using Bayesian inference
        
        The process continues until convergence, providing both refined estimates and uncertainty
        quantification through the learned covariance structure.
        
        Mathematical Framework:
            The EM algorithm maximizes the log-likelihood of the observed data by iteratively improving 
            estimates of both the latent states (log NAV values) and model parameters (covariance matrix).
            
            State Evolution Model:
                x_t = x_{t-1} + mu_t + epsilon_t,
                where epsilon_t ~ N(0, Sigma)
            
            Observation Model:
                y_t = H_t x_t + eta_t,
                where eta_t ~ N(0, R_t)
            
            The algorithm uses a Wishart prior on the inverse covariance to ensure positive definiteness 
            and provide regularization when data is sparse.
        
        Args:
            data_input (EstimationInputs):
                Structured input containing:
                    - Valuation data for the private company (sparse, with measurement uncertainty)
                    - Market cap data for comparable public companies (frequent observations)
                    - Cashflow series affecting private company value
                    - Parameter specifications (volatilities, correlations, growth rates)
            
            cov_private_and_comps (np.ndarray):
                Initial covariance matrix estimate of shape (1+num_comparables, 1+num_comparables) 
                governing the joint dynamics between private company and comparables:
                    - [0,0]: Private company variance
                    - [0,1:]: Covariances between private company and comparables
                    - [1:,1:]: Covariance matrix among comparable companies
            
            initial_private_estimate (pd.Series):
                Initial NAV estimates for the private company indexed by date.
                Used as starting point for the iterative estimation process.
                Shape: (num_time_periods,)
        
        Returns:
            tuple[pd.Series, np.ndarray]:
                - nav_estimate (pd.Series): Final optimized NAV estimates for the private company
                  indexed by date, incorporating both data fitting and learned dynamics
                - final_cov_private_and_comps (np.ndarray): Converged dynamics covariance matrix 
                  that best explains the observed joint evolution of all companies
                    - [0,0]: Private company variance
                    - [0,1:] and [1:,0]: Cross-covariances with comparables
                    - [1:,1:]: Covariance matrix among comparable companies
        
        Raises:
            ValueError: If EM_num_iters is not a positive integer
            ValueError: If EM_error_thresh is not a positive float
            ValueError: If the estimated covariance gets NaNs or Infs
        
        Algorithm Details:
            1. **Initialization**: Start with prior covariance and initial state estimates
            2. **E-step**: Given current Σ, solve MAP optimization for optimal state sequence
            3. **M-step**: Given current states, update Σ using empirical + theoretical covariance
            4. **Convergence Check**: Compare successive covariance estimates using infinity norm
            5. **Repeat**: Continue until convergence or maximum iterations reached
            
            The M-step combines three components:
                - Empirical covariance: Sample covariance of estimated disturbances
                - Theoretical covariance: Expected covariance from posterior uncertainty
                - Prior covariance: Wishart prior for regularization
        
        Notes:
            - Uses Wishart prior distribution for the inverse covariance matrix
            - Theoretical covariance is computed from the Hessian of the MAP objective
            - Convergence is measured by relative change in covariance matrix elements
            - Each iteration refines both state estimates and uncertainty quantification
            - The final result provides both point estimates and learned model parameters
        """
        
        # ====================================================================
        # Algorithm Configuration and Validation
        # ====================================================================
        
        num_iters = self.EM_num_iters
        error_thresh = self.EM_error_thresh
        
        if not isinstance(num_iters, int) or num_iters < 1:
            raise ValueError("Number of EM iterations needs to be a positive integer")
        
        if not isinstance(error_thresh, float) or error_thresh <= 0:
            raise ValueError("EM error threshold must be a positive float.")
        
        # ====================================================================
        # Problem Dimensions and Setup
        # ====================================================================
        
        number_of_dates = len(data_input.data.dates)  # Number of time steps
        num_priva_and_comps = (
            1 + data_input.data.number_of_comparables
        )  # Number of private and public comps
        
        # ====================================================================
        # Bayesian Prior Specification
        # ====================================================================
        
        # We use a Wishart prior distribution for the inverse covariance matrix
        # Reference: https://en.wikipedia.org/wiki/Wishart_distribution
        #
        # For inverse covariance Lambda = Sigma^(-1), we specify:
        #   Lambda ~ Wishart(nu, V^(-1)) where:
        #     - nu = degrees of freedom (set to n+2 for minimal informativeness)
        #     - V = scale matrix (set to prior covariance estimate)
        #
        # This ensures positive definiteness and provides regularization
        # when empirical estimates are unreliable due to sparse data
        
        # V parameter for Wishart prior (our prior belief about the covariance)
        wishart_scale_matrix = cov_private_and_comps
        
        # ====================================================================
        # EM Algorithm Initialization
        # ====================================================================
        
        error = np.inf  # Initial error (will be updated after first iteration)
        iteration = 0  # Iteration counter
        
        # Initial E-step: Compute optimal state sequence given initial covariance
        # This provides the starting point for the EM iterations
        nav_estimate, disturbance_estimate = self._point_estimate(
            data_input=data_input,
            cov_private_and_comps=cov_private_and_comps,
            initial_private_estimate=initial_private_estimate,
        )
        
        # ====================================================================
        # Main EM Iteration Loop
        # ====================================================================
        
        while error >= error_thresh and iteration < num_iters:
            # Store previous estimate for convergence checking
            prev_cov_private_and_comps = cov_private_and_comps
            
            # Extract dynamic disturbances from the current state estimate
            # d_dyn[company, time] = estimated innovation for company between time t and t+1
            # These represent the unexplained changes not captured by deterministic dynamics
            d_dyn = disturbance_estimate.T
            
            # ================================================================
            # M-Step: Update Covariance Matrix
            # ================================================================
            
            # The M-step combines three sources of information to update the covariance:
            
            # 1. Empirical Covariance: Sample covariance of estimated disturbances
            #    This captures the observed variability in the dynamic innovations
            empirical_covariance = d_dyn @ d_dyn.T
            
            # 2. Theoretical Covariance: Expected posterior covariance from uncertainty
            #    This accounts for the uncertainty in our state estimates
            _hessian = self._compute_hessian_of_objective(
                data_input=data_input, cov_private_and_comps=cov_private_and_comps
            )
            
            # Posterior covariance is the inverse of the Hessian for a Gaussian model
            covariance = np.linalg.inv(_hessian)
            
            # Extract the contribution to dynamics covariance from estimation uncertainty
            # We reshape the full covariance matrix to access time-company block structure
            reshaped_covariance = covariance.reshape(
                number_of_dates, num_priva_and_comps, number_of_dates, num_priva_and_comps
            )
            
            # Compute the theoretical contribution to dynamics covariance
            # This involves summing diagonal blocks and subtracting off-diagonal blocks
            # to isolate the covariance of dynamic disturbances
            sum_block_diags = (
                reshaped_covariance[range(number_of_dates), :, range(number_of_dates), :].sum(
                    axis=0
                )
                # Correct for double-counting of boundary terms (first and last periods)
                # These periods only couple to one neighbor, not two
                - reshaped_covariance[0, :, 0, :]
                - reshaped_covariance[-1, :, -1, :]
            )
            
            # Sum off-diagonal contributions (coupling between consecutive periods)
            sum_off_diagonal_blocks = reshaped_covariance[
                range(1, number_of_dates), :, range(number_of_dates - 1), :
            ].sum(axis=0)
            
            # Final theoretical covariance contribution
            # This represents the additional uncertainty from estimation errors
            theoretical_covariance = (
                sum_block_diags - sum_off_diagonal_blocks - sum_off_diagonal_blocks.T
            )
            
            # 3. Combine all components using Bayesian update formula
            #    The division by number_of_dates implements the proper Bayesian averaging 
            #    for the Wishart posterior with T observations plus prior
            cov_private_and_comps = (
                empirical_covariance + theoretical_covariance + wishart_scale_matrix
            ) / number_of_dates
            
            # ================================================================
            # Convergence Assessment
            # ================================================================
            
            # Check convergence using relative change in covariance matrix
            # We use the infinity norm (maximum absolute element) for robustness
            error = np.linalg.norm(
                (cov_private_and_comps - prev_cov_private_and_comps).reshape(-1), ord=np.inf
            ) / np.median(np.diag(cov_private_and_comps))
            
            iteration += 1
            
            if not np.isfinite(cov_private_and_comps).all():
                raise ValueError(
                    "NaNs or Inf detected in the covariance matrix during Expectation Maximization algorithm"
                )
            
            # ================================================================
            # E-Step: Update State Estimates
            # ================================================================
            
            # Given the updated covariance matrix, recompute optimal state sequence
            # This completes one full EM iteration
            nav_estimate, disturbance_estimate = self._point_estimate(
                data_input=data_input,
                cov_private_and_comps=cov_private_and_comps,
                initial_private_estimate=nav_estimate,
            )
        
        return nav_estimate, cov_private_and_comps
    
    def __call__(self, data_input: EstimationInputs) -> Estimate:
        """Execute the complete MLE Gaussian Fusion estimation process.
        
        This is the main entry point for the estimator that orchestrates the entire private company 
        NAV estimation process. The method combines sparse private company valuations with frequent 
        public comparable observations to produce optimal NAV estimates with uncertainty quantification.
        
        The estimation process follows these key steps:
            1. **Initialization**: Set up problem dimensions and compute initial estimates
            2. **Covariance Construction**: Build joint covariance matrix for dynamics
            3. **Method Selection**: Choose between fixed or adaptive covariance estimation
            4. **Point Estimation**: Solve MAP optimization for optimal state sequence
            5. **Uncertainty Quantification**: Compute posterior covariance from Hessian
            6. **Result Packaging**: Format estimates into standardized output structure
        
        Mathematical Framework:
            The estimator implements a Bayesian state-space model where:
                - **State Evolution**: x_t = x_{t-1} + mu_t + epsilon_t
                - **Private Observations**: y_t^private = x_t^private + eta_t^private
                - **Comparable Observations**: y_t^comps = x_t^comps + eta_t^comps
                - **Joint Dynamics**: [x_t^private, x_t^comps] coupled through Sigma_dynamics
        
        Args:
            data_input (EstimationInputs):
                Complete structured input containing:
                    - Valuation data for the private company (sparse observations)
                    - Market cap data for comparable public companies (frequent observations)
                    - Cashflow series affecting private company value
                    - All necessary parameters (volatilities, correlations, growth rates)
        
        Returns:
            Estimate:
                Standardized estimation results containing:
                    - **nav_estimate**: DataFrame with columns ['central', 'lower', 'upper']
                        * 'central': Point estimates of NAV indexed by date
                        * 'lower': Lower confidence bound (geometric standard deviation)
                        * 'upper': Upper confidence bound (geometric standard deviation)
        
        Notes:
            - **Single time period**: Returns initial estimator result directly (no dynamics)
            - **No comparables**: Estimation relies purely on private company data
            - **Sparse observations**: Method handles missing valuations gracefully
        
        Example Usage:
            ```python
            estimator = NAVcasting(dyn_tune="EM")
            result = estimator(data_input)
            nav_time_series = result.nav_estimate['central']
            ```
        
        Notes:
            - The method automatically handles missing observations in valuation data
            - Uncertainty quantification accounts for both observation noise and model uncertainty
            - Results are always in original dollar units (not log-space)
            - The estimator maintains numerical stability through eigenvalue regularization
        """
        
        # ====================================================================
        # Algorithm Initialization and Setup
        # ====================================================================
        
        # Reset solver state to ensure clean execution
        # This prevents issues from previous runs affecting current estimation
        self.solver_point_estimate = None
        
        # Extract problem dimensions from input data
        # These dimensions drive memory allocation and computational complexity
        number_of_dates = len(data_input.data.dates)  # T: number of time periods
        num_priva_and_comps = (
            1 + data_input.data.number_of_comparables
        )  # n: private company + public comparables
        
        # ====================================================================
        # Initial Estimate Computation
        # ====================================================================
        
        # Compute baseline estimate using the configured initial estimator
        # This provides a starting point and fallback for single-period problems
        # The initial estimator typically uses simple interpolation or averaging methods
        initial_estimate = self.initial_estimator(data_input=data_input).nav_estimate["central"]
        
        # ====================================================================
        # Dynamics Covariance Matrix Construction
        # ====================================================================
        
        # Build the joint covariance matrix that governs the evolution of
        # private company value and public comparable market caps
        # This captures correlations and volatilities for regularization
        cov_private_and_comps = self._compute_cov_private_and_comps(data_input=data_input)
        
        # ====================================================================
        # Handle Edge Case: Single Time Period
        # ====================================================================
        
        # For single-period problems, no dynamics are involved
        # Return the initial estimate directly to avoid unnecessary computation
        if number_of_dates == 1:
            return self.initial_estimator(data_input=data_input)
        
        # ====================================================================
        # Covariance Matrix Estimation Strategy
        # ====================================================================
        
        # Choose between fixed covariance (faster) and adaptive EM-based refinement (more accurate)
        if self.method_dyn_tune == "none":
            # **Fixed Covariance Approach**
            # Use the pre-computed covariance matrix without further refinement
            # Suitable when user has high confidence in input parameters
            # or when computational efficiency is prioritized over precision
            point_estimate, _ = self._point_estimate(
                data_input=data_input,
                cov_private_and_comps=cov_private_and_comps,
                initial_private_estimate=initial_estimate,
            )
        
        elif self.method_dyn_tune == "EM":
            # **Adaptive Covariance Approach**
            # Use Expectation-Maximization to iteratively refine both:
            #   1. State estimates (NAV time series)
            #   2. Covariance matrix (dynamics parameters)
            # This approach learns optimal parameters from the data itself
            point_estimate, cov_private_and_comps = self._tune_dynamics_covar_with_EM(
                data_input=data_input,
                cov_private_and_comps=cov_private_and_comps,
                initial_private_estimate=initial_estimate,
            )
        
        else:
            # Invalid method specification - provide clear error message
            raise ValueError("Dynamics covariance matrix tuning method not recognized.")
        
        # ====================================================================
        # Posterior Uncertainty Quantification
        # ====================================================================
        
        # Compute the posterior covariance matrix by inverting the Hessian
        # The Hessian represents the curvature of the log-likelihood at the MAP estimate
        # Its inverse gives us the posterior covariance (uncertainty quantification)
        covariance = np.linalg.inv(
            self._compute_hessian_of_objective(
                data_input=data_input,
                cov_private_and_comps=cov_private_and_comps,
            )
        )
        
        # Extract standard deviations for the private company only
        # The covariance matrix has block structure: [private, comparables] × [time]
        # We extract diagonal elements corresponding to private company states
        # The step size num_priva_and_comps skips comparable companies in the extraction
        private_std = np.sqrt(np.diag(covariance)[::num_priva_and_comps])
        
        # ====================================================================
        # Result Packaging and Standardization
        # ====================================================================
        
        # Convert the raw estimates into the standardized output format
        # This ensures consistent API regardless of internal computational details
        nav_estimate = pd.DataFrame(
            data=0.0, index=data_input.data.dates, columns=["central", "lower", "upper"]
        )
        
        nav_estimate["central"] = point_estimate
        nav_estimate["lower"] = point_estimate * np.exp(-private_std)
        nav_estimate["upper"] = point_estimate * np.exp(private_std)
        
        # Return wrapped result in the Estimate dataclass
        # This provides type safety and validation for the output
        return Estimate(nav_estimate=nav_estimate)


# ############################################################################
# Example Usage and Testing
# ############################################################################

if __name__ == "__main__":
    
    def build_mock_estimation_inputs(
        seed: int = 42,
        n_comparables: int = 3,
    ) -> EstimationInputs:
        """Create a mock EstimationData instance that satisfies all validation rules.
        
        Rules enforced:
            - valuations.index == cashflows.index == comparables.index
            - cashflow == 0 on any date where at least one valuation is non-NaN
            - comparables_longer_horizon extends further back in time with same columns
        """
        rng = np.random.default_rng(seed)
        
        # Core date range (monthly)
        dates = pd.date_range("2023-01-31", periods=7, freq="MS")
        
        # Create two valuation sources with alternating availability (NaNs create room for cashflows)
        valuations = pd.DataFrame(
            {
                "NAV valuation 1": [100.0, np.nan, 105.0, np.nan, 116.0, np.nan, 112.0],
                "NAV valuation 2": [98.0, np.nan, 103.0, np.nan, np.nan, np.nan, 108.0],
            },
            index=dates,
        )
        
        # Cashflows
        cashflows = pd.Series(
            [0.0, 0.0, 0.0, 10.0, 0.0, -5.0, 0.0],
            index=dates,
        )
        
        # Comparable companies (same index as valuations)
        comp_cols = [f"Comparable {i + 1}" for i in range(n_comparables)]
        comparables = pd.DataFrame(
            rng.normal(loc=100, scale=5, size=(len(dates), n_comparables)).round(2),
            index=dates,
            columns=comp_cols,
        )
        
        # Extended history for comparables (earlier 4 months)
        extended_dates = pd.date_range(dates[0] - pd.offsets.MonthEnd(4), periods=4, freq="MS")
        comparables_longer_horizon = pd.DataFrame(
            rng.normal(loc=80, scale=4, size=(len(extended_dates), n_comparables)).round(2),
            index=extended_dates,
            columns=comp_cols,
        )
        
        estimation_data = EstimationData(
            valuations=valuations,
            cashflows=cashflows,
            comparables=comparables,
            comparables_longer_horizon=comparables_longer_horizon,
        )
        
        estimation_parameters = EstimationLogscaleParameters(
            valuations_std=pd.Series({"NAV valuation 1": 0.1, "NAV valuation 2": 0.2}),
            volatility=0.1,
            correlations_to_private=pd.Series(
                {"Comparable 1": 0.7, "Comparable 2": 0.5, "Comparable 3": 0.6}
            ),
            comparables_covariance=None,
        )
        
        return EstimationInputs(data=estimation_data, parameters=estimation_parameters)
    
    # Build mock data
    mock = build_mock_estimation_inputs()
    
    # Test RollForward
    roll_forward = RollForward()(mock)
    
    # Test NAVcasting with EM
    estimator_gauss = NAVcasting(dyn_tune="EM")
    
    # Test individual methods
    cov_private_and_comps = estimator_gauss._compute_cov_private_and_comps(data_input=mock)
    
    hessian = estimator_gauss._compute_hessian_of_objective(
        data_input=mock, cov_private_and_comps=cov_private_and_comps
    )
    
    point_estimate = estimator_gauss._point_estimate(
        data_input=mock,
        cov_private_and_comps=cov_private_and_comps,
        initial_private_estimate=roll_forward.nav_estimate["central"],
    )
    
    EM = estimator_gauss._tune_dynamics_covar_with_EM(
        data_input=mock,
        cov_private_and_comps=cov_private_and_comps,
        initial_private_estimate=roll_forward.nav_estimate["central"],
    )
