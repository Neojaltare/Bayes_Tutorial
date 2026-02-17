"""
Streamlit App: Frequentist vs Bayesian Regression
Interactive teaching tool for research audiences

Demonstrates:
- Frequentist point estimates and confidence intervals
- Bayesian prior, likelihood, and posterior
- Credible intervals vs confidence intervals
- Bayes factors and model comparison
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Frequentist vs Bayesian Regression",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better formatting
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .interpretation-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #262730;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_data(n, alpha_true, beta_true, sigma, seed=None):
    """Generate synthetic regression data: y = alpha + beta * x + noise"""
    if seed is not None:
        np.random.seed(seed)
    x = np.random.uniform(0, 6, n)
    noise = np.random.normal(0, sigma, n)
    y = alpha_true + beta_true * x + noise
    return x, y

def frequentist_regression(x, y, sigma):
    """
    Compute frequentist OLS estimate and confidence interval for both intercept and slope
    
    Returns:
    - alpha_hat: OLS intercept estimate
    - beta_hat: OLS slope estimate
    - se_alpha: standard error of intercept
    - se_beta: standard error of slope
    - ci_alpha_lower, ci_alpha_upper: 95% CI for intercept
    - ci_beta_lower, ci_beta_upper: 95% CI for slope
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # OLS slope estimate
    beta_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
    
    # OLS intercept estimate
    alpha_hat = y_mean - beta_hat * x_mean
    
    # Standard errors
    # For slope
    se_beta = sigma / np.sqrt(np.sum((x - x_mean)**2))
    
    # For intercept
    se_alpha = sigma * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))
    
    # 95% confidence intervals
    ci_beta_lower = beta_hat - 1.96 * se_beta
    ci_beta_upper = beta_hat + 1.96 * se_beta
    
    ci_alpha_lower = alpha_hat - 1.96 * se_alpha
    ci_alpha_upper = alpha_hat + 1.96 * se_alpha
    
    return alpha_hat, beta_hat, se_alpha, se_beta, ci_alpha_lower, ci_alpha_upper, ci_beta_lower, ci_beta_upper

def bayesian_regression(x, y, sigma, mu_alpha, tau_alpha, mu_beta, tau_beta):
    """
    Compute Bayesian posterior using conjugate normal-normal model with intercept
    
    Prior: alpha ~ N(mu_alpha, tau_alpha^2), beta ~ N(mu_beta, tau_beta^2)
    Likelihood: y_i | alpha, beta ~ N(alpha + beta * x_i, sigma^2)
    Posterior: (alpha, beta) | y ~ N(mu_n, Sigma_n)
    
    Returns:
    - mu_alpha_n: posterior mean for intercept
    - mu_beta_n: posterior mean for slope
    - tau_alpha_n: posterior standard deviation for intercept
    - tau_beta_n: posterior standard deviation for slope
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # For computational simplicity, we can use centered parameterization
    # This decouples intercept and slope posteriors
    
    # Slope posterior (same as before but with centered x)
    x_centered = x - x_mean
    precision_beta_prior = 1 / tau_beta**2
    precision_beta_likelihood = np.sum(x_centered**2) / sigma**2
    precision_beta_posterior = precision_beta_prior + precision_beta_likelihood
    tau_beta_n_squared = 1 / precision_beta_posterior
    tau_beta_n = np.sqrt(tau_beta_n_squared)
    
    mu_beta_n = tau_beta_n_squared * (mu_beta * precision_beta_prior + 
                                       np.sum(x_centered * (y - y_mean)) / sigma**2)
    
    # Intercept posterior (conditional on slope being integrated out)
    # Simplified: treating as independent for conjugacy
    precision_alpha_prior = 1 / tau_alpha**2
    precision_alpha_likelihood = n / sigma**2
    precision_alpha_posterior = precision_alpha_prior + precision_alpha_likelihood
    tau_alpha_n_squared = 1 / precision_alpha_posterior
    tau_alpha_n = np.sqrt(tau_alpha_n_squared)
    
    # Posterior mean for intercept
    y_adjusted = y_mean - mu_beta_n * x_mean
    mu_alpha_n = tau_alpha_n_squared * (mu_alpha * precision_alpha_prior + 
                                         y_adjusted * n / sigma**2)
    
    return mu_alpha_n, mu_beta_n, tau_alpha_n, tau_beta_n

def compute_likelihood(x, y, alpha_grid, beta_grid, sigma):
    """
    Compute likelihood as a function of alpha and beta (for visualization)
    For plotting, we'll compute marginal likelihoods
    
    Returns normalized likelihood for plotting
    """
    # For slope (marginalizing over intercept)
    likelihood_beta = np.ones_like(beta_grid)
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    for i, beta_val in enumerate(beta_grid):
        # MLE for alpha given beta
        alpha_mle = y_mean - beta_val * x_mean
        log_lik = np.sum(stats.norm.logpdf(y, loc=alpha_mle + beta_val * x, scale=sigma))
        likelihood_beta[i] = np.exp(log_lik)
    
    # Normalize for visualization
    likelihood_beta = likelihood_beta / np.max(likelihood_beta)
    
    # For intercept (marginalizing over slope)
    likelihood_alpha = np.ones_like(alpha_grid)
    for i, alpha_val in enumerate(alpha_grid):
        # MLE for beta given alpha
        beta_mle = np.sum((x - x_mean) * (y - alpha_val)) / np.sum((x - x_mean)**2)
        log_lik = np.sum(stats.norm.logpdf(y, loc=alpha_val + beta_mle * x, scale=sigma))
        likelihood_alpha[i] = np.exp(log_lik)
    
    # Normalize for visualization
    likelihood_alpha = likelihood_alpha / np.max(likelihood_alpha)
    
    return likelihood_alpha, likelihood_beta

def compute_bayes_factor(x, y, sigma, mu_alpha, tau_alpha, mu_beta, tau_beta, beta_null=0):
    """
    Compute Bayes Factor comparing:
    H1: beta ~ N(mu_beta, tau_beta^2) (alternative model with prior on beta)
    H0: beta = beta_null (null model, e.g., beta = 0)
    
    BF_{10} = p(y | H1) / p(y | H0)
    
    Uses marginal likelihood (evidence) for each model
    Both models have intercept integrated out (or equivalently, vague prior on intercept)
    """
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    
    # For H1: Marginal likelihood with normal prior on beta
    # After centering, beta and alpha are approximately independent
    # Marginal likelihood: integral over beta of p(y|beta) * p(beta)
    
    # Using centered data (removes intercept from the problem)
    y_centered = y - y_mean
    
    # Predictive distribution for y_centered given prior on beta:
    # y_centered ~ N(mu_beta * x_centered, sigma^2 * I + tau_beta^2 * x_centered * x_centered^T)
    # For computational simplicity, we compute this as a product of independent normals
    # This is exact for the marginal likelihood of beta
    
    # Variance of each y_i (centered) under the prior predictive
    predictive_var_h1 = sigma**2 + tau_beta**2 * x_centered**2
    
    # Log marginal likelihood under H1
    log_ml_h1 = np.sum(stats.norm.logpdf(y_centered, loc=mu_beta * x_centered, 
                                          scale=np.sqrt(predictive_var_h1)))
    
    # Under H0 (fixed beta = beta_null)
    # Marginal likelihood with beta fixed (intercept still integrated out)
    # Using centered data: y_centered ~ N(beta_null * x_centered, sigma^2)
    log_ml_h0 = np.sum(stats.norm.logpdf(y_centered, loc=beta_null * x_centered, scale=sigma))
    
    # Bayes Factor (on log scale)
    log_bf_10 = log_ml_h1 - log_ml_h0
    bf_10 = np.exp(log_bf_10)
    
    return bf_10, log_bf_10

# ============================================================================
# SIDEBAR: USER INPUTS
# ============================================================================

st.sidebar.title("Configuration")

st.sidebar.header("Data Generation")
n = st.sidebar.slider("Sample size (n)", 5, 100, 50, 1)
alpha_true = st.sidebar.slider("True intercept (α_true)", -4.0, 4.0, 0.0, 0.1)
beta_true = st.sidebar.slider("True slope (β_true)", -4.0, 4.0, 1.0, 0.1)
sigma = st.sidebar.slider("Noise std dev (σ)", 0.1, 6.0, 1.0, 0.1)

# Random seed for reproducibility
if 'data_seed' not in st.session_state:
    st.session_state.data_seed = 42

if st.sidebar.button("Regenerate Data"):
    st.session_state.data_seed = np.random.randint(0, 10000)
st.sidebar.markdown("<div style='margin-bottom:150px;'></div>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("Bayesian Prior - Slope")
mu_beta = st.sidebar.slider("Prior mean for slope (μ_β)", -4.0, 4.0, 0.0, 0.1)
tau_beta = st.sidebar.slider("Prior std dev for slope (τ_β)", 0.1, 6.0, 1.0, 0.1)
st.sidebar.markdown("<div style='margin-bottom:70px;'></div>", unsafe_allow_html=True)

st.sidebar.header("Bayesian Prior - Intercept")
mu_alpha = st.sidebar.slider("Prior mean for intercept (μ_α)", -4.0, 4.0, 0.0, 0.1)
tau_alpha = st.sidebar.slider("Prior std dev for intercept (τ_α)", 0.1, 6.0, 1.0, 0.1)
st.sidebar.markdown("<div style='margin-bottom:150px;'></div>", unsafe_allow_html=True)



# -----------------------------
# Sliders for likelihood parameters
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("## Parameters for the Likelihood")
alpha_slider = st.sidebar.slider("Intercept (α)", -5.0, 5.0, 0.0)
beta_slider = st.sidebar.slider("Slope (β)", -5.0, 5.0, 0.0)
st.sidebar.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)




# Add sidebar controls for example prior distributions
st.sidebar.markdown("---")
st.sidebar.header("Example Prior Parameters")

# Gaussian parameters
st.sidebar.subheader("Gaussian")
gauss_mean = st.sidebar.slider("Gaussian mean", -4.0, 4.0, 0.0, 0.1, key="gauss_mean")
gauss_std = st.sidebar.slider("Gaussian std", 0.1, 6.0, 1.0, 0.1, key="gauss_std")

# Uniform parameters
st.sidebar.subheader("Uniform")
uniform_lower = st.sidebar.slider("Uniform lower", -5.0, 0.0, -2.0, 0.1, key="unif_low")
uniform_upper = st.sidebar.slider("Uniform upper", 0.0, 5.0, 2.0, 0.1, key="unif_up")

# Beta parameters
st.sidebar.subheader("Beta")
beta_a = st.sidebar.slider("Beta α", 0.5, 10.0, 2.0, 0.5, key="beta_a")
beta_b = st.sidebar.slider("Beta β", 0.5, 10.0, 5.0, 0.5, key="beta_b")

# Student-t parameters
st.sidebar.subheader("Student-t")
t_df = st.sidebar.slider("t degrees of freedom", 1, 30, 3, 1, key="t_df")
t_scale = st.sidebar.slider("t scale", 0.1, 3.0, 1.0, 0.1, key="t_scale")

# Cauchy parameters
st.sidebar.subheader("Cauchy")
cauchy_scale = st.sidebar.slider("Cauchy scale", 0.1, 2.0, 0.5, 0.1, key="cauchy_scale")

# Laplace parameters
st.sidebar.subheader("Laplace")
laplace_scale = st.sidebar.slider("Laplace scale", 0.1, 2.0, 0.5, 0.1, key="laplace_scale")

# Create a fixed grid for all prior examples
prior_example_grid = np.linspace(-5, 5, 500)

st.sidebar.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)


st.sidebar.markdown("---")
st.sidebar.header("Model Comparison - Bayes Factor")
beta_null = st.sidebar.slider("Null hypothesis value (β₀)", -4.0, 4.0, 0.0, 0.1)

# ============================================================================
# GENERATE DATA
# ============================================================================

x, y = generate_data(n, alpha_true, beta_true, sigma, seed=st.session_state.data_seed)

# Compute frequentist estimates
alpha_hat, beta_hat, se_alpha, se_beta, ci_alpha_lower, ci_alpha_upper, ci_beta_lower, ci_beta_upper = frequentist_regression(x, y, sigma)

# Compute Bayesian posterior
mu_alpha_n, mu_beta_n, tau_alpha_n, tau_beta_n = bayesian_regression(x, y, sigma, mu_alpha, tau_alpha, mu_beta, tau_beta)

# Posterior credible intervals (95%)
credible_alpha_lower = mu_alpha_n - 1.96 * tau_alpha_n
credible_alpha_upper = mu_alpha_n + 1.96 * tau_alpha_n
credible_beta_lower = mu_beta_n - 1.96 * tau_beta_n
credible_beta_upper = mu_beta_n + 1.96 * tau_beta_n

# Probability that beta > 0
prob_beta_positive = 1 - stats.norm.cdf(0, loc=mu_beta_n, scale=tau_beta_n)

# Compute Bayes Factor
bf_10, log_bf_10 = compute_bayes_factor(x, y, sigma, mu_alpha, tau_alpha, mu_beta, tau_beta, beta_null)

# ============================================================================
# MAIN APP
# ============================================================================

st.title("Frequentist vs Bayesian Regression")
st.markdown("### *An Interactive Teaching Tool for Bayesian Inference*")

st.markdown(f"""
**Scenario:** We observe {n} data points from the model: 
**y = α + β·x + ε**, where **ε ~ N(0, {sigma}²)**

We want to learn about the regression coefficients **α** (intercept) and **β** (slope).
""")

# ============================================================================
# SECTION A: FREQUENTIST APPROACH
# ============================================================================

st.markdown("---")
st.markdown("## The Frequentist Approach")

st.markdown("""
The frequentist framework gives us:
- **Point estimates** (α̂, β̂) via Ordinary Least Squares (OLS)
- **Confidence intervals** based on the sampling distribution
- But **no probability distribution** over the parameters themselves
""")

col1, col2 = st.columns([1.5, 1])

with col1:
    # Plot 1: Data with regression line
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    # Plot vertical lines from each point to the fitted line (residuals)
    y_fitted = alpha_hat + beta_hat * x
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y[i], y_fitted[i]], 'r-', alpha=0.3, linewidth=1)
    
    # Scatter plot
    ax1.scatter(x, y, alpha=0.6, s=50, label='Observed data', zorder=3)
    
    # Fitted line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = alpha_hat + beta_hat * x_line
    ax1.plot(x_line, y_line, 'b-', linewidth=2.5, label=f'Fitted line (α̂={alpha_hat:.2f}, β̂={beta_hat:.3f})', zorder=2)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Frequentist Regression: Point Estimates', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with col2:
    # Plot 2: Coefficient estimates with CI (both slope and intercept)
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(6, 8))
    
    # SLOPE
    ax2a.errorbar([0], [beta_hat], yerr=[[beta_hat - ci_beta_lower], [ci_beta_upper - beta_hat]], 
                 fmt='o', markersize=12, capsize=10, capthick=2, 
                 color='darkblue', ecolor='navy', linewidth=2.5)
    
    ax2a.axhline(beta_true, color='red', linestyle='--', linewidth=2, 
                label=f'True β = {beta_true:.2f}', alpha=0.7)
    
    ax2a.set_xlim(-0.5, 0.5)
    y_range_beta = max(4*se_beta, abs(beta_hat - beta_true) + 2*se_beta)
    ax2a.set_ylim(beta_true - y_range_beta, beta_true + y_range_beta)
    ax2a.set_ylabel('β (Slope)', fontsize=14)
    ax2a.set_title('Slope Estimate + 95% CI', fontsize=12, fontweight='bold')
    ax2a.set_xticks([])
    ax2a.legend(fontsize=10)
    ax2a.grid(True, alpha=0.3, axis='y')
    ax2a.text(0.02, beta_hat, f'  β̂ = {beta_hat:.3f}', 
             fontsize=10, va='center', fontweight='bold')
    
    # INTERCEPT
    ax2b.errorbar([0], [alpha_hat], yerr=[[alpha_hat - ci_alpha_lower], [ci_alpha_upper - alpha_hat]], 
                 fmt='o', markersize=12, capsize=10, capthick=2, 
                 color='darkgreen', ecolor='green', linewidth=2.5)
    
    ax2b.axhline(alpha_true, color='red', linestyle='--', linewidth=2, 
                label=f'True α = {alpha_true:.2f}', alpha=0.7)
    
    ax2b.set_xlim(-0.5, 0.5)
    y_range_alpha = max(4*se_alpha, abs(alpha_hat - alpha_true) + 2*se_alpha)
    ax2b.set_ylim(alpha_true - y_range_alpha, alpha_true + y_range_alpha)
    ax2b.set_ylabel('α (Intercept)', fontsize=14)
    ax2b.set_title('Intercept Estimate + 95% CI', fontsize=12, fontweight='bold')
    ax2b.set_xticks([])
    ax2b.legend(fontsize=10)
    ax2b.grid(True, alpha=0.3, axis='y')
    ax2b.text(0.02, alpha_hat, f'  α̂ = {alpha_hat:.3f}', 
             fontsize=10, va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig2)

# Interpretation box
st.markdown(f"""
<div class="interpretation-box">
<b>Frequentist Results:</b><br>
• <b>Intercept estimate:</b> α̂ = {alpha_hat:.3f} ± {se_alpha:.3f}<br>
• <b>Slope estimate:</b> β̂ = {beta_hat:.3f} ± {se_beta:.3f}<br>
• <b>95% CI for slope:</b> [{ci_beta_lower:.3f}, {ci_beta_upper:.3f}]<br>
• <b>95% CI for intercept:</b> [{ci_alpha_lower:.3f}, {ci_alpha_upper:.3f}]<br><br>

<b>What the CI means:</b><br>
If we repeated this experiment many times, 95% of the confidence intervals constructed this way would contain the true parameters.<br><br>

<b>What it does NOT mean:</b><br>
It is <b>not</b> a probability statement about the parameters. We cannot say "there's a 95% probability that β is in this interval."<br>
In frequentist statistics, parameters are fixed (unknown) values, not random variables.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SECTION B: TRANSITION TO BAYESIAN
# ============================================================================

st.markdown("---")
st.markdown("## Why Go Bayesian?")

st.markdown("""
<div style="font-size:25px; line-height:1.6;">
The frequentist approach has limitations:

1. <b>No uncertainty quantification for parameters</b>: We get point estimates and CIs, but no distribution over parameters  
2. <b>Cannot make probability statements</b>: "What's the probability that β > 0?" — can't answer this!  
3. <b>No principled way to incorporate prior knowledge</b>: If we have domain expertise, how do we use it?  
4. <b>Limited ability to compare models</b>: How do we quantify evidence for one hypothesis vs another?  

<b>The Bayesian solution:</b> Treat parameters as random variables with probability distributions.  
This lets us:
- Express uncertainty explicitly through <b>posterior distributions</b>  
- Make direct probability statements about parameters  
- Update beliefs as we see data (<b>prior → posterior</b>)  
- More principled way to compare models using <b>Bayes factors</b>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SECTION C: BAYESIAN APPROACH
# ============================================================================

# Create grids for plotting distributions
beta_grid = np.linspace(
    min(mu_beta - 4*tau_beta, mu_beta_n - 4*tau_beta_n, beta_hat - 4*se_beta),
    max(mu_beta + 4*tau_beta, mu_beta_n + 4*tau_beta_n, beta_hat + 4*se_beta),
    500
)

alpha_grid = np.linspace(
    min(mu_alpha - 4*tau_alpha, mu_alpha_n - 4*tau_alpha_n, alpha_hat - 4*se_alpha),
    max(mu_alpha + 4*tau_alpha, mu_alpha_n + 4*tau_alpha_n, alpha_hat + 4*se_alpha),
    500
)

# Compute prior, likelihood, and posterior densities
prior_beta = stats.norm.pdf(beta_grid, loc=mu_beta, scale=tau_beta)
prior_alpha = stats.norm.pdf(alpha_grid, loc=mu_alpha, scale=tau_alpha)

likelihood_alpha, likelihood_beta = compute_likelihood(x, y, alpha_grid, beta_grid, sigma)

posterior_beta = stats.norm.pdf(beta_grid, loc=mu_beta_n, scale=tau_beta_n)
posterior_alpha = stats.norm.pdf(alpha_grid, loc=mu_alpha_n, scale=tau_alpha_n)

st.markdown("<div style='margin-bottom:100px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## The Bayesian Approach")

# Create two plots side by side for slope and intercept - COMBINED VIEW
st.markdown("### There are three key components to the Bayesian approach:")

st.markdown("### Prior → Likelihood → Posterior")


st.markdown("<div style='margin-bottom:100px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## Understanding the Prior")

st.markdown(f"""
**Bayesian model for the same data:**

- **Priors:** 
  - α ~ N({mu_alpha}, {tau_alpha}²) — initial beliefs about intercept
  - β ~ N({mu_beta}, {tau_beta}²) — initial beliefs about slope
""")


# ============================================================================
# SEPARATED PLOTS: PRIOR, LIKELIHOOD, POSTERIOR
# ============================================================================

st.markdown("---")
# ROW 1: PRIORS
st.markdown("**Prior Distributions** — Our initial beliefs before seeing data")
col_p1, col_p2 = st.columns(2)

with col_p1:
    fig_p1, ax_p1 = plt.subplots(figsize=(7, 5))
    ax_p1.plot(beta_grid, prior_beta, 'b-', linewidth=3)
    ax_p1.axvline(mu_beta, color='darkblue', linestyle='--', linewidth=2, alpha=0.7)
    ax_p1.axvline(beta_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True β = {beta_true:.2f}')
    ax_p1.fill_between(beta_grid, 0, prior_beta, color='blue', alpha=0.2)
    ax_p1.set_xlabel('β (Slope)', fontsize=12)
    ax_p1.set_ylabel('Density', fontsize=12)
    ax_p1.set_title(f'Prior for Slope: N({mu_beta}, {tau_beta}²)', fontsize=12, fontweight='bold')
    ax_p1.text(mu_beta, ax_p1.get_ylim()[1]*0.9, f'  Prior Mean = {mu_beta}', 
               fontsize=10, color='darkblue', fontweight='bold')
    ax_p1.set_xlim(-5, 5)  # Fixed x-axis
    ax_p1.legend(fontsize=10)
    ax_p1.grid(True, alpha=0.3)
    st.pyplot(fig_p1)

with col_p2:
    fig_p2, ax_p2 = plt.subplots(figsize=(7, 5))
    ax_p2.plot(alpha_grid, prior_alpha, 'b-', linewidth=3)
    ax_p2.axvline(mu_alpha, color='darkblue', linestyle='--', linewidth=2, alpha=0.7)
    ax_p2.axvline(alpha_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True α = {alpha_true:.2f}')
    ax_p2.fill_between(alpha_grid, 0, prior_alpha, color='blue', alpha=0.2)
    ax_p2.set_xlabel('α (Intercept)', fontsize=12)
    ax_p2.set_ylabel('Density', fontsize=12)
    ax_p2.set_title(f'Prior for Intercept: N({mu_alpha}, {tau_alpha}²)', fontsize=12, fontweight='bold')
    ax_p2.text(mu_alpha, ax_p2.get_ylim()[1]*0.9, f'  Prior Mean = {mu_alpha}', 
               fontsize=10, color='darkblue', fontweight='bold')
    ax_p2.set_xlim(-5, 5)  # Fixed x-axis
    ax_p2.legend(fontsize=10)
    ax_p2.grid(True, alpha=0.3)
    st.pyplot(fig_p2)


# Add this as a new subsection right after the frequentist section or before the Bayesian section
st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## Understanding the Likelihood")

st.markdown(f"""
- **Likelihood:** y | α, β ~ N(α + β·x, {sigma}²) — same data model as frequentist approach

- The **likelihood function** describes how probable our observed data is for different parameter values.

In our model, each observation y_i is assumed to come from a normal distribution:
- **Mean:** α + β·x_i (the regression line at that x value)
- **Standard deviation:** σ = {sigma}

Below, we visualize this: each data point has a normal distribution "attached" to it, centered on the regression line.
""")

# fig_lik, ax_lik = plt.subplots(figsize=(12, 8))

# # Scatter plot
# ax_lik.scatter(x, y, alpha=0.8, s=80, color='red', zorder=5, label='Observed data', edgecolors='darkred', linewidth=1.5)

# # Fitted line
# x_line = np.linspace(x.min(), x.max(), 100)
# y_line_freq = alpha_hat + beta_hat * x_line
# ax_lik.plot(x_line, y_line_freq, 'b-', linewidth=3, label=f'Fitted line (α̂={alpha_hat:.2f}, β̂={beta_hat:.3f})', zorder=4)

# # For each data point, draw a normal distribution centered on the predicted value
# # We'll plot these as curves perpendicular to the x-axis
# n_points_to_show = min(len(x), 15)  # Limit to avoid overcrowding
# indices_to_show = np.linspace(0, len(x)-1, n_points_to_show, dtype=int)

# for idx in indices_to_show:
#     x_val = x[idx]
#     y_pred = alpha_hat + beta_hat * x_val  # Predicted mean at this x
    
#     # Create a normal distribution centered at y_pred with std sigma
#     y_range = np.linspace(y_pred - 3*sigma, y_pred + 3*sigma, 100)
#     density = stats.norm.pdf(y_range, loc=y_pred, scale=sigma)
    
#     # Scale the density for visualization (make it wider so it's visible)
#     # We'll plot it in the x-direction, scaled appropriately
#     x_scale = 0.3 * (x.max() - x.min()) / n_points_to_show  # Adjust scaling
#     x_offset = density * x_scale
    
#     # Plot the normal curve as a function going horizontally from the regression line
#     ax_lik.plot(x_val + x_offset, y_range, 'gray', alpha=0.5, linewidth=1.5)
#     ax_lik.fill_betweenx(y_range, x_val, x_val + x_offset, color='lightblue', alpha=0.3)
    
#     # Draw a line from the data point to the regression line (residual)
#     ax_lik.plot([x_val, x_val], [y[idx], y_pred], 'r--', alpha=0.4, linewidth=1)

# ax_lik.set_xlabel('x', fontsize=14)
# ax_lik.set_ylabel('y', fontsize=14)
# ax_lik.set_title('Likelihood Visualization: Each Point Has a Normal Distribution', fontsize=15, fontweight='bold')
# ax_lik.legend(fontsize=12, loc='best')
# ax_lik.grid(True, alpha=0.3)

# st.pyplot(fig_lik)



# -----------------------------
# Compute log-likelihood
# -----------------------------
y_pred = alpha_slider + beta_slider * x
residuals = y - y_pred
log_likelihood = np.sum(stats.norm.logpdf(residuals, loc=0, scale=sigma))
y_center = np.mean(y)
half_range = (max(y) - min(y)) / 2
half_range *= 3  

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter data
ax.scatter(x, y, alpha=0.8, s=70, color='red',
           edgecolors='darkred', linewidth=1.5,
           label='Observed data')

# Regression line from sliders
x_line = np.linspace(x.min(), x.max(), 200)
y_line = alpha_slider + beta_slider * x_line
ax.plot(x_line, y_line, 'b-', linewidth=3,
        label=f'Line (α={alpha_slider:.2f}, β={beta_slider:.2f})')

# Draw conditional normal distributions
n_points_to_show = min(len(x), 15)
indices_to_show = np.linspace(0, len(x)-1,
                              n_points_to_show,
                              dtype=int)

for idx in indices_to_show:
    x_val = x[idx]
    y_pred_point = alpha_slider + beta_slider * x_val

    y_range = np.linspace(y_pred_point - 3*sigma,
                          y_pred_point + 3*sigma,
                          100)
    density = stats.norm.pdf(y_range,
                             loc=y_pred_point,
                             scale=sigma)

    # Scale for visualization
    x_scale = 0.3 * (x.max() - x.min()) / n_points_to_show
    x_offset = density * x_scale

    ax.plot(x_val + x_offset, y_range,
            'gray', alpha=0.6, linewidth=1.2)
    ax.fill_betweenx(y_range,
                     x_val,
                     x_val + x_offset,
                     color='lightblue',
                     alpha=0.3)

    # # Residual line
    # ax.plot([x_val, x_val],
    #         [y[idx], y_pred_point],
    #         'r--', alpha=0.4)

ax.set_xlim(0, 6.5)
ax.set_ylim(y_center - half_range, y_center + half_range)
ax.set_xlabel('x', fontsize=13)
ax.set_ylabel('y', fontsize=13)
ax.set_title('Likelihood Visualization: Parameters Move, Data Stay Fixed',
             fontsize=15, fontweight='bold')

ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# -----------------------------
# Display likelihood value
# -----------------------------
st.markdown("### Likelihood Information")
st.write(f"Log-Likelihood: {log_likelihood:.2f}")

st.markdown("""
**Key idea:**  
The data are fixed.  
We change α and β.  
The likelihood tells us how compatible this parameter choice is with the observed data.
""")







st.markdown(f"""
<div class="interpretation-box">
<b>What This Shows:</b><br>
- The <b>blue line</b> is our regression line (predicted mean for each x)<br>
- Each <b>gray/blue bell curve</b> shows the normal distribution for that x value<br>
- The actual <b>red data points</b> are random draws from these distributions<br>
- The <b>red dashed lines</b> show residuals (distance from observation to predicted mean)<br><br>

<b>Key Insight:</b><br>
The likelihood L(α, β | data) is the product of all these normal densities evaluated at the observed y values.
Points close to the line (small residuals) contribute higher probability, while points far from the line contribute lower probability.<br><br>

This is why the likelihood peaks near the OLS estimates — they minimize the sum of squared residuals!
</div>
""", unsafe_allow_html=True)

# ROW 2: LIKELIHOODS
st.markdown("**Likelihood Functions** — What the data tells us (normalized for visualization)")

# First show the marginal likelihoods
col_l1, col_l2 = st.columns(2)

with col_l1:
    fig_l1, ax_l1 = plt.subplots(figsize=(7, 5))
    ax_l1.plot(beta_grid, likelihood_beta, 'orange', linewidth=3)
    ax_l1.axvline(beta_hat, color='darkorange', linestyle='--', linewidth=2, alpha=0.7)
    ax_l1.axvline(beta_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True β = {beta_true:.2f}')
    ax_l1.fill_between(beta_grid, 0, likelihood_beta, color='orange', alpha=0.2)
    ax_l1.set_xlabel('β (Slope)', fontsize=12)
    ax_l1.set_ylabel('Normalized Likelihood', fontsize=12)
    ax_l1.set_title('Marginal Likelihood for Slope', fontsize=12, fontweight='bold')
    ax_l1.text(beta_hat, ax_l1.get_ylim()[1]*0.9, f'  MLE = {beta_hat:.3f}', 
               fontsize=10, color='darkorange', fontweight='bold')
    ax_l1.legend(fontsize=10)
    ax_l1.grid(True, alpha=0.3)
    st.pyplot(fig_l1)

with col_l2:
    fig_l2, ax_l2 = plt.subplots(figsize=(7, 5))
    ax_l2.plot(alpha_grid, likelihood_alpha, 'orange', linewidth=3)
    ax_l2.axvline(alpha_hat, color='darkorange', linestyle='--', linewidth=2, alpha=0.7)
    ax_l2.axvline(alpha_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True α = {alpha_true:.2f}')
    ax_l2.fill_between(alpha_grid, 0, likelihood_alpha, color='orange', alpha=0.2)
    ax_l2.set_xlabel('α (Intercept)', fontsize=12)
    ax_l2.set_ylabel('Normalized Likelihood', fontsize=12)
    ax_l2.set_title('Marginal Likelihood for Intercept', fontsize=12, fontweight='bold')
    ax_l2.text(alpha_hat, ax_l2.get_ylim()[1]*0.9, f'  MLE = {alpha_hat:.3f}', 
               fontsize=10, color='darkorange', fontweight='bold')
    ax_l2.legend(fontsize=10)
    ax_l2.grid(True, alpha=0.3)
    st.pyplot(fig_l2)

# NOW ADD THE 2D JOINT LIKELIHOOD SURFACE
st.markdown("**Joint Likelihood Surface** — Likelihood over both parameters simultaneously")

# Compute joint likelihood on a 2D grid
alpha_grid_2d = np.linspace(alpha_hat - 3*se_alpha, alpha_hat + 3*se_alpha, 100)
beta_grid_2d = np.linspace(beta_hat - 3*se_beta, beta_hat + 3*se_beta, 100)
Alpha_mesh, Beta_mesh = np.meshgrid(alpha_grid_2d, beta_grid_2d)

# Compute log-likelihood for each (alpha, beta) pair
log_likelihood_surface = np.zeros_like(Alpha_mesh)
for i in range(Alpha_mesh.shape[0]):
    for j in range(Alpha_mesh.shape[1]):
        alpha_val = Alpha_mesh[i, j]
        beta_val = Beta_mesh[i, j]
        # Compute log-likelihood
        y_pred = alpha_val + beta_val * x
        log_lik = np.sum(stats.norm.logpdf(y, loc=y_pred, scale=sigma))
        log_likelihood_surface[i, j] = log_lik

# Convert to likelihood (normalized for visualization)
likelihood_surface = np.exp(log_likelihood_surface - np.max(log_likelihood_surface))

# Create figure with two subplots: contour and 3D surface
fig_joint = plt.figure(figsize=(16, 6))

# SUBPLOT 1: Contour plot
ax_contour = fig_joint.add_subplot(121)
contour = ax_contour.contourf(Alpha_mesh, Beta_mesh, likelihood_surface, levels=20, cmap='YlOrRd')
contour_lines = ax_contour.contour(Alpha_mesh, Beta_mesh, likelihood_surface, levels=10, colors='black', alpha=0.3, linewidths=0.5)

# Mark MLE
ax_contour.plot(alpha_hat, beta_hat, 'r*', markersize=20, label=f'MLE (α̂={alpha_hat:.2f}, β̂={beta_hat:.2f})', zorder=5)
# Mark true values
ax_contour.plot(alpha_true, beta_true, 'bo', markersize=12, label=f'True (α={alpha_true:.2f}, β={beta_true:.2f})', zorder=5)

ax_contour.set_xlabel('α (Intercept)', fontsize=13)
ax_contour.set_ylabel('β (Slope)', fontsize=13)
ax_contour.set_title('Joint Likelihood: Contour Plot', fontsize=14, fontweight='bold')
ax_contour.legend(fontsize=11)
ax_contour.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax_contour, label='Normalized Likelihood')

# SUBPLOT 2: 3D surface plot
ax_3d = fig_joint.add_subplot(122, projection='3d')
surf = ax_3d.plot_surface(Alpha_mesh, Beta_mesh, likelihood_surface, cmap='YlOrRd', alpha=0.8, edgecolor='none')

# Mark MLE on 3D plot
ax_3d.scatter([alpha_hat], [beta_hat], [1.0], color='red', s=100, marker='*', label='MLE', zorder=10)
ax_3d.scatter([alpha_true], [beta_true], [np.exp(np.sum(stats.norm.logpdf(y, loc=alpha_true + beta_true * x, scale=sigma)) - np.max(log_likelihood_surface))], 
              color='blue', s=80, marker='o', label='True', zorder=10)

ax_3d.set_xlabel('α (Intercept)', fontsize=11)
ax_3d.set_ylabel('β (Slope)', fontsize=11)
ax_3d.set_zlabel('Normalized Likelihood', fontsize=11)
ax_3d.set_title('Joint Likelihood: 3D Surface', fontsize=14, fontweight='bold')
ax_3d.view_init(elev=25, azim=45)
ax_3d.legend(fontsize=10)

plt.tight_layout()
st.pyplot(fig_joint)

st.markdown(f"""
<div class="interpretation-box">
<b>Understanding the Joint Likelihood Surface:</b><br><br>
<b>Left (Contour Plot):</b><br>
- Shows likelihood as a "topographic map" over (α, β) space<br>
- Each contour line represents points with equal likelihood<br>
- The <b>red star</b> marks the MLE — the peak of the likelihood<br>
- The <b>blue circle</b> marks the true parameter values<br><br>
<b>Right (3D Surface):</b><br>
- Same information as a 3D "hill"<br>
- Height represents how likely that (α, β) combination is given the data<br>
- The peak occurs at the MLE estimates<br><br>
<b>Key Insights:</b><br>
- The likelihood "ridge" shows the correlation between α and β<br>
- The shape shows which parameter combinations are plausible given the data<br>
- Narrower ridges = more information from data; wider = more uncertainty<br>
- This is what gets combined with the prior to form the posterior!
</div>
""", unsafe_allow_html=True)

# ROW 3: POSTERIORS
st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## Understanding the Posterior")
st.markdown("**Posterior Distributions** — Updated beliefs after combining prior and likelihood")

st.markdown(f"""
- **Posteriors:** 
  - α | y ~ N({mu_alpha_n:.3f}, {tau_alpha_n:.3f}²) — updated beliefs about intercept
  - β | y ~ N({mu_beta_n:.3f}, {tau_beta_n:.3f}²) — updated beliefs about slope
The posterior combines prior knowledge with observed data using **Bayes' theorem**.
""")

col_po1, col_po2 = st.columns(2)

with col_po1:
    fig_po1, ax_po1 = plt.subplots(figsize=(7, 5))
    ax_po1.plot(beta_grid, posterior_beta, 'g-', linewidth=3)
    ax_po1.axvline(mu_beta_n, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax_po1.axvline(beta_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True β = {beta_true:.2f}')
    
    # Shade 95% credible interval
    ci_mask = (beta_grid >= credible_beta_lower) & (beta_grid <= credible_beta_upper)
    ax_po1.fill_between(beta_grid[ci_mask], 0, posterior_beta[ci_mask], 
                        color='green', alpha=0.3, label='95% Credible Interval')
    
    ax_po1.set_xlabel('β (Slope)', fontsize=12)
    ax_po1.set_ylabel('Density', fontsize=12)
    ax_po1.set_title(f'Posterior for Slope: N({mu_beta_n:.3f}, {tau_beta_n:.3f}²)', fontsize=12, fontweight='bold')
    ax_po1.text(mu_beta_n, ax_po1.get_ylim()[1]*0.9, f'  Post. Mean = {mu_beta_n:.3f}', 
                fontsize=10, color='darkgreen', fontweight='bold')
    ax_po1.legend(fontsize=10)
    ax_po1.grid(True, alpha=0.3)
    st.pyplot(fig_po1)

with col_po2:
    fig_po2, ax_po2 = plt.subplots(figsize=(7, 5))
    ax_po2.plot(alpha_grid, posterior_alpha, 'purple', linewidth=3)
    ax_po2.axvline(mu_alpha_n, color='darkviolet', linestyle='--', linewidth=2, alpha=0.7)
    ax_po2.axvline(alpha_true, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'True α = {alpha_true:.2f}')
    
    # Shade 95% credible interval
    ci_mask_alpha = (alpha_grid >= credible_alpha_lower) & (alpha_grid <= credible_alpha_upper)
    ax_po2.fill_between(alpha_grid[ci_mask_alpha], 0, posterior_alpha[ci_mask_alpha], 
                        color='purple', alpha=0.3, label='95% Credible Interval')
    
    ax_po2.set_xlabel('α (Intercept)', fontsize=12)
    ax_po2.set_ylabel('Density', fontsize=12)
    ax_po2.set_title(f'Posterior for Intercept: N({mu_alpha_n:.3f}, {tau_alpha_n:.3f}²)', fontsize=12, fontweight='bold')
    ax_po2.text(mu_alpha_n, ax_po2.get_ylim()[1]*0.9, f'  Post. Mean = {mu_alpha_n:.3f}', 
                fontsize=10, color='darkviolet', fontweight='bold')
    ax_po2.legend(fontsize=10)
    ax_po2.grid(True, alpha=0.3)
    st.pyplot(fig_po2)


# ============================================================================
# BAYESIAN VISUALIZATIONS
# ============================================================================
st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)

st.markdown("### Bayesian Inference: Putting it together")

# Create two plots side by side for slope and intercept - COMBINED VIEW
st.markdown("#### Prior → Likelihood → Posterior for Slope and Intercept")

col_beta, col_alpha = st.columns(2)

with col_beta:
    # Plot for SLOPE
    fig3, ax3 = plt.subplots(figsize=(7, 6))
    
    ax3.plot(beta_grid, prior_beta, 'b-', linewidth=2.5, label=f'Prior: N({mu_beta}, {tau_beta}²)', alpha=0.8)
    ax3.plot(beta_grid, likelihood_beta, 'orange', linewidth=2.5, label='Likelihood (normalized)', alpha=0.8)
    ax3.plot(beta_grid, posterior_beta, 'g-', linewidth=3, label=f'Posterior: N({mu_beta_n:.3f}, {tau_beta_n:.3f}²)')
    
    # Mark posterior mean
    ax3.axvline(mu_beta_n, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax3.text(mu_beta_n, ax3.get_ylim()[1]*0.95, f'  Post. Mean = {mu_beta_n:.3f}', 
             rotation=0, va='top', fontsize=9, color='darkgreen', fontweight='bold')
    
    # Mark true value
    ax3.axvline(beta_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax3.text(beta_true, ax3.get_ylim()[1]*0.85, f'  True β = {beta_true:.2f}', 
             rotation=0, va='top', fontsize=9, color='red', fontweight='bold')
    
    # Shade 95% credible interval
    ci_mask = (beta_grid >= credible_beta_lower) & (beta_grid <= credible_beta_upper)
    ax3.fill_between(beta_grid[ci_mask], 0, posterior_beta[ci_mask], 
                      color='green', alpha=0.2, label='95% Credible Interval')
    
    ax3.set_xlabel('β (Slope)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Slope: Bayesian Updating', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with col_alpha:
    # Plot for INTERCEPT
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    
    ax4.plot(alpha_grid, prior_alpha, 'b-', linewidth=2.5, label=f'Prior: N({mu_alpha}, {tau_alpha}²)', alpha=0.8)
    ax4.plot(alpha_grid, likelihood_alpha, 'orange', linewidth=2.5, label='Likelihood (normalized)', alpha=0.8)
    ax4.plot(alpha_grid, posterior_alpha, 'purple', linewidth=3, label=f'Posterior: N({mu_alpha_n:.3f}, {tau_alpha_n:.3f}²)')
    
    # Mark posterior mean
    ax4.axvline(mu_alpha_n, color='darkviolet', linestyle='--', linewidth=2, alpha=0.7)
    ax4.text(mu_alpha_n, ax4.get_ylim()[1]*0.95, f'  Post. Mean = {mu_alpha_n:.3f}', 
             rotation=0, va='top', fontsize=9, color='darkviolet', fontweight='bold')
    
    # Mark true value
    ax4.axvline(alpha_true, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax4.text(alpha_true, ax4.get_ylim()[1]*0.85, f'  True α = {alpha_true:.2f}', 
             rotation=0, va='top', fontsize=9, color='red', fontweight='bold')
    
    # Shade 95% credible interval
    ci_mask_alpha = (alpha_grid >= credible_alpha_lower) & (alpha_grid <= credible_alpha_upper)
    ax4.fill_between(alpha_grid[ci_mask_alpha], 0, posterior_alpha[ci_mask_alpha], 
                      color='purple', alpha=0.2, label='95% Credible Interval')
    
    ax4.set_xlabel('α (Intercept)', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Intercept: Bayesian Updating', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)


st.markdown(f"""
<div class="interpretation-box">
<b>Understanding the Plots:</b><br>
• <b>Blue (Prior):</b> Our initial beliefs before seeing data.<br>
• <b>Orange (Likelihood):</b> What the data tells us. Peaks near the MLE (α̂ = {alpha_hat:.3f}, β̂ = {beta_hat:.3f}).<br>
• <b>Green/Purple (Posterior):</b> Our updated beliefs after seeing data. A compromise between prior and likelihood.<br><br>

<b>Key insight:</b> The posterior is a <b>weighted average</b> of prior and data:<br>
• With more data (larger n), likelihood dominates → posterior ≈ MLE<br>
• With less data, prior has more influence → posterior pulled toward prior mean<br>
• This is <b>shrinkage</b> or <b>regularization</b> in action!
</div>
""", unsafe_allow_html=True)





st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)

st.markdown("""
## Bayes' Theorem: The Foundation of Bayesian Inference

At the heart of Bayesian statistics is **Bayes' theorem**, which tells us how to update our beliefs based on evidence.
""")

# Display Bayes' theorem in both forms
col_bayes1, col_bayes2 = st.columns(2)

with col_bayes1:
    st.markdown("""
    **Formal Statement:**
    """)
    st.latex(r"""
    p(\theta | y) = \frac{p(y | \theta) \cdot p(\theta)}{p(y)}
    """)
    st.markdown("""
    Where:
    - $p(\\theta | y)$ = **Posterior** (updated beliefs)
    - $p(y | \\theta)$ = **Likelihood** (data given parameters)
    - $p(\\theta)$ = **Prior** (initial beliefs)
    - $p(y)$ = **Marginal likelihood** (normalizing constant)
    """)

with col_bayes2:
    st.markdown("""
    **Intuitive Form:**
    """)
    st.latex(r"""
    \text{Posterior} \propto \text{Prior} \times \text{Likelihood}
    """)
    st.markdown("""
    Or in words:
    
    **Updated beliefs = Initial beliefs × Evidence from data**
    
    The $\\propto$ symbol means "proportional to" — we don't need the exact normalizing constant to understand how beliefs update.
    """)




# ============================================================================
# COMPARISON OF REGRESSION LINES
# ============================================================================
st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)
st.markdown("### Comparing Frequentist vs Bayesian Estimates")

col3, col4 = st.columns([1.5, 1])

with col3:
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.scatter(x, y, alpha=0.6, s=50, color='gray', label='Observed data')
    
    # Frequentist line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_freq = alpha_hat + beta_hat * x_line
    ax5.plot(x_line, y_freq, 'b-', linewidth=2.5, 
             label=f'Frequentist (α̂={alpha_hat:.2f}, β̂={beta_hat:.3f})', alpha=0.7)
    
    # Bayesian line
    y_bayes = mu_alpha_n + mu_beta_n * x_line
    ax5.plot(x_line, y_bayes, 'g-', linewidth=2.5, 
             label=f'Bayesian (E[α|y]={mu_alpha_n:.2f}, E[β|y]={mu_beta_n:.3f})', alpha=0.8)
    
    # True line
    y_true = alpha_true + beta_true * x_line
    ax5.plot(x_line, y_true, 'r--', linewidth=2, 
             label=f'True (α={alpha_true:.2f}, β={beta_true:.2f})', alpha=0.7)
    
    ax5.set_xlabel('x', fontsize=12)
    ax5.set_ylabel('y', fontsize=12)
    ax5.set_title('Regression Lines: Frequentist vs Bayesian', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)

with col4:
    # Summary statistics comparison
    st.markdown("#### Estimate Comparison")
    
    comparison_df = pd.DataFrame({
        'Parameter': ['Intercept (α)', 'Intercept (α)', 'Slope (β)', 'Slope (β)'],
        'Method': ['Frequentist', 'Bayesian', 'Frequentist', 'Bayesian'],
        'Estimate': [f'{alpha_hat:.4f}', f'{mu_alpha_n:.4f}', f'{beta_hat:.4f}', f'{mu_beta_n:.4f}'],
        'Std Error/Dev': [f'{se_alpha:.4f}', f'{tau_alpha_n:.4f}', f'{se_beta:.4f}', f'{tau_beta_n:.4f}'],
        '95% Interval': [
            f'[{ci_alpha_lower:.3f}, {ci_alpha_upper:.3f}]', 
            f'[{credible_alpha_lower:.3f}, {credible_alpha_upper:.3f}]',
            f'[{ci_beta_lower:.3f}, {ci_beta_upper:.3f}]', 
            f'[{credible_beta_lower:.3f}, {credible_beta_upper:.3f}]'
        ]
    })
    
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)
    
    st.markdown(f"""
    **Differences:**
    - Intercept: {abs(alpha_hat - mu_alpha_n):.4f}
    - Slope: {abs(beta_hat - mu_beta_n):.4f}
    
    <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border-left: 5px solid #ff9800; margin-top: 10px;">
    <b style="color: #000000; font-size: 16px;">Shrinkage Effect:</b> <span style="color: #000000; font-weight: bold;">Bayesian estimates are pulled toward prior means, especially with small n. 
    This is regularization in action — the prior acts as a penalty that prevents overfitting!</span>
    </div>
    """, unsafe_allow_html=True)



# ROW 2: Comparison of impact on posterior
st.markdown("<div style='margin-bottom:150px;'></div>", unsafe_allow_html=True)
st.markdown("## Impact of Priors on Posterior Estimates")

st.markdown("""
Let's see how these different priors affect the posterior when combined with the same data.
Assume we observe data that suggests β ≈ 1.5 with standard error 0.4.
""")

# Simulate a likelihood centered at 1.5
beta_mle_example = 1.5
se_example = 0.4
likelihood_example = stats.norm.pdf(prior_example_grid, loc=beta_mle_example, scale=se_example)
likelihood_example = likelihood_example / np.max(likelihood_example)  # Normalize for visualization

# Compute posteriors for each prior type
# Informative prior
prior_info = stats.norm(loc=1.0, scale=0.3)
posterior_info_mean = (1.0/(0.3**2) + beta_mle_example/(se_example**2)) / (1/(0.3**2) + 1/(se_example**2))
posterior_info_var = 1 / (1/(0.3**2) + 1/(se_example**2))
posterior_info = stats.norm.pdf(prior_example_grid, loc=posterior_info_mean, scale=np.sqrt(posterior_info_var))

# Weakly informative prior
prior_weak = stats.norm(loc=0.0, scale=1.0)
posterior_weak_mean = (0.0/(1.0**2) + beta_mle_example/(se_example**2)) / (1/(1.0**2) + 1/(se_example**2))
posterior_weak_var = 1 / (1/(1.0**2) + 1/(se_example**2))
posterior_weak = stats.norm.pdf(prior_example_grid, loc=posterior_weak_mean, scale=np.sqrt(posterior_weak_var))

# Uninformative prior
prior_unin = stats.norm(loc=0.0, scale=10.0)
posterior_unin_mean = (0.0/(10.0**2) + beta_mle_example/(se_example**2)) / (1/(10.0**2) + 1/(se_example**2))
posterior_unin_var = 1 / (1/(10.0**2) + 1/(se_example**2))
posterior_unin = stats.norm.pdf(prior_example_grid, loc=posterior_unin_mean, scale=np.sqrt(posterior_unin_var))

fig_comp, (ax_comp1, ax_comp2, ax_comp3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Informative
ax_comp1.plot(prior_example_grid, stats.norm.pdf(prior_example_grid, 1.0, 0.3), 'b--', linewidth=2, label='Prior: N(1.0, 0.3²)', alpha=0.7)
ax_comp1.plot(prior_example_grid, likelihood_example, 'orange', linewidth=2, label=f'Likelihood (MLE={beta_mle_example})', alpha=0.7)
ax_comp1.plot(prior_example_grid, posterior_info, 'g-', linewidth=3, label=f'Posterior: N({posterior_info_mean:.2f}, {np.sqrt(posterior_info_var):.2f}²)')
ax_comp1.axvline(posterior_info_mean, color='darkgreen', linestyle='--', alpha=0.5)
ax_comp1.axvline(beta_mle_example, color='orange', linestyle=':', alpha=0.5)
ax_comp1.set_xlabel('β', fontsize=12)
ax_comp1.set_ylabel('Density', fontsize=12)
ax_comp1.set_title('Informative Prior\nPosterior pulled toward prior', fontsize=12, fontweight='bold')
ax_comp1.set_xlim(-1, 3)
ax_comp1.legend(fontsize=9)
ax_comp1.grid(True, alpha=0.3)

# Plot 2: Weakly Informative
ax_comp2.plot(prior_example_grid, stats.norm.pdf(prior_example_grid, 0.0, 1.0), 'b--', linewidth=2, label='Prior: N(0.0, 1.0²)', alpha=0.7)
ax_comp2.plot(prior_example_grid, likelihood_example, 'orange', linewidth=2, label=f'Likelihood (MLE={beta_mle_example})', alpha=0.7)
ax_comp2.plot(prior_example_grid, posterior_weak, 'g-', linewidth=3, label=f'Posterior: N({posterior_weak_mean:.2f}, {np.sqrt(posterior_weak_var):.2f}²)')
ax_comp2.axvline(posterior_weak_mean, color='darkgreen', linestyle='--', alpha=0.5)
ax_comp2.axvline(beta_mle_example, color='orange', linestyle=':', alpha=0.5)
ax_comp2.set_xlabel('β', fontsize=12)
ax_comp2.set_ylabel('Density', fontsize=12)
ax_comp2.set_title('Weakly Informative Prior\nBalanced compromise', fontsize=12, fontweight='bold')
ax_comp2.set_xlim(-1, 3)
ax_comp2.legend(fontsize=9)
ax_comp2.grid(True, alpha=0.3)

# Plot 3: Uninformative
ax_comp3.plot(prior_example_grid, stats.norm.pdf(prior_example_grid, 0.0, 10.0), 'b--', linewidth=2, label='Prior: N(0.0, 10.0²)', alpha=0.7)
ax_comp3.plot(prior_example_grid, likelihood_example, 'orange', linewidth=2, label=f'Likelihood (MLE={beta_mle_example})', alpha=0.7)
ax_comp3.plot(prior_example_grid, posterior_unin, 'g-', linewidth=3, label=f'Posterior: N({posterior_unin_mean:.2f}, {np.sqrt(posterior_unin_var):.2f}²)')
ax_comp3.axvline(posterior_unin_mean, color='darkgreen', linestyle='--', alpha=0.5)
ax_comp3.axvline(beta_mle_example, color='orange', linestyle=':', alpha=0.5)
ax_comp3.set_xlabel('β', fontsize=12)
ax_comp3.set_ylabel('Density', fontsize=12)
ax_comp3.set_title('Uninformative Prior\nPosterior ≈ Likelihood', fontsize=12, fontweight='bold')
ax_comp3.set_xlim(-1, 3)
ax_comp3.legend(fontsize=9)
ax_comp3.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_comp)


st.markdown(f"""
<div class="interpretation-box">
<b>Key Takeaways:</b><br><br>

<b>Informative Prior (τ = 0.3):</b><br>
- Strong belief pulls posterior away from data (MLE = {beta_mle_example})<br>
- Posterior mean = {posterior_info_mean:.3f} (closer to prior mean of 1.0)<br>
- Use when you have strong domain knowledge<br><br>

<b>Weakly Informative Prior (τ = 1.0):</b><br>
- Balanced between prior and data<br>
- Posterior mean = {posterior_weak_mean:.3f} (compromise)<br>
- Good default: regularizes without overwhelming data<br><br>

<b>Uninformative Prior (τ = 10.0):</b><br>
- Flat/vague prior → data dominates<br>
- Posterior mean = {posterior_unin_mean:.3f} (≈ MLE)<br>
- Equivalent to frequentist inference when variance → ∞<br><br>

<b>Practical Advice:</b> Start with weakly informative priors. They provide gentle regularization 
without imposing strong beliefs, and are robust to prior misspecification.
</div>
""", unsafe_allow_html=True)







# ============================================================================
# BAYESIAN INFERENCE: PROBABILITY STATEMENTS
# ============================================================================
st.markdown("<div style='margin-bottom:150px;'></div>", unsafe_allow_html=True)
st.markdown("### Making Probability Statements (Bayesian Advantage)")

col5, col6 = st.columns(2)

with col5:
    st.markdown(f"""
    <div class="interpretation-box">
    <b>95% Credible Interval for Slope (β):</b><br>
    [{credible_beta_lower:.3f}, {credible_beta_upper:.3f}]<br><br>
    
    <b>This DOES mean:</b><br>
    "Given the data we observed, there is a 95% probability that β lies in this interval."<br><br>
    
    This is a direct probability statement about the parameter — something we <b>cannot</b> do in frequentist statistics!
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="interpretation-box">
    <b>Probability that β > {beta_null}:</b><br>
    <span style="font-size:24px; font-weight:bold; color:darkgreen;">
    P(β > {beta_null} | data) = {prob_beta_positive:.4f}
    </span><br><br>
    
    <b>Interpretation:</b><br>
    Based on our posterior beliefs, there is a {prob_beta_positive*100:.2f}% chance that the slope is greater than {beta_null}.
    </div>
    """, unsafe_allow_html=True)


# ROW 1: Show different types of prior distributions
st.markdown("<div style='margin-bottom:170px;'></div>", unsafe_allow_html=True)

st.markdown("### Common Prior Distributions")

col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    # GAUSSIAN/NORMAL PRIOR
    fig_gauss, ax_gauss = plt.subplots(figsize=(6, 5))
    
    gaussian_prior = stats.norm.pdf(prior_example_grid, loc=gauss_mean, scale=gauss_std)
    ax_gauss.plot(prior_example_grid, gaussian_prior, 'darkgreen', linewidth=3, label=f'N({gauss_mean}, {gauss_std}²)')
    ax_gauss.fill_between(prior_example_grid, 0, gaussian_prior, color='green', alpha=0.3)
    ax_gauss.axvline(gauss_mean, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    
    ax_gauss.set_xlabel('Parameter Value', fontsize=11)
    ax_gauss.set_ylabel('Density', fontsize=11)
    ax_gauss.set_title('Gaussian (Normal) Prior', fontsize=12, fontweight='bold', color='darkgreen')
    ax_gauss.set_xlim(-5, 5)
    ax_gauss.set_ylim(0, None)
    ax_gauss.legend(fontsize=10)
    ax_gauss.grid(True, alpha=0.3)
    st.pyplot(fig_gauss)
    
    st.markdown("""
    **Gaussian/Normal:**
    - Continuous, unbounded
    - Bell-shaped, symmetric
    - Conjugate for regression
    - Common for location parameters
    - Easy to interpret (mean ± SD)
    """)

with col_info2:
    # UNIFORM PRIOR
    fig_unif, ax_unif = plt.subplots(figsize=(6, 5))
    
    # Uniform distribution
    uniform_prior = np.zeros_like(prior_example_grid)
    uniform_height = 1 / (uniform_upper - uniform_lower) if uniform_upper > uniform_lower else 0
    uniform_prior[(prior_example_grid >= uniform_lower) & (prior_example_grid <= uniform_upper)] = uniform_height
    ax_unif.plot(prior_example_grid, uniform_prior, 'orange', linewidth=3, label=f'Uniform({uniform_lower}, {uniform_upper})')
    ax_unif.fill_between(prior_example_grid, 0, uniform_prior, color='orange', alpha=0.3)
    
    ax_unif.set_xlabel('Parameter Value', fontsize=11)
    ax_unif.set_ylabel('Density', fontsize=11)
    ax_unif.set_title('Uniform Prior', fontsize=12, fontweight='bold', color='darkorange')
    ax_unif.set_xlim(-5, 5)
    ax_unif.set_ylim(0, max(uniform_height * 1.2, 0.5))
    ax_unif.legend(fontsize=10)
    ax_unif.grid(True, alpha=0.3)
    st.pyplot(fig_unif)
    
    st.markdown("""
    **Uniform:**
    - Flat over bounded interval
    - "Non-informative" (equal weight)
    - Bounded support
    - Simple but can be improper
    - Use when no prior knowledge
    """)

with col_info3:
    # BETA PRIOR
    fig_beta, ax_beta = plt.subplots(figsize=(6, 5))
    
    # Beta distribution (map to 0-1, then scale for display)
    beta_grid = np.linspace(0, 1, 500)
    beta_prior = stats.beta.pdf(beta_grid, a=beta_a, b=beta_b)
    ax_beta.plot(beta_grid, beta_prior, 'purple', linewidth=3, label=f'Beta({beta_a}, {beta_b})')
    ax_beta.fill_between(beta_grid, 0, beta_prior, color='purple', alpha=0.3)
    
    ax_beta.set_xlabel('Parameter Value', fontsize=11)
    ax_beta.set_ylabel('Density', fontsize=11)
    ax_beta.set_title('Beta Prior', fontsize=12, fontweight='bold', color='purple')
    ax_beta.set_xlim(0, 1)
    ax_beta.set_ylim(0, None)
    ax_beta.legend(fontsize=10)
    ax_beta.grid(True, alpha=0.3)
    st.pyplot(fig_beta)
    
    st.markdown("""
    **Beta:**
    - Bounded: [0, 1]
    - Flexible shapes (α, β parameters)
    - Conjugate for proportions
    - Common for probabilities
    - Can be uniform when α=β=1
    """)

# Add a second row with more distributions
st.markdown("### Additional Prior Distributions")

col_info4, col_info5, col_info6 = st.columns(3)

with col_info4:
    # STUDENT-T PRIOR (heavy tails)
    fig_t, ax_t = plt.subplots(figsize=(6, 5))
    
    t_prior = stats.t.pdf(prior_example_grid, df=t_df, loc=0, scale=t_scale)
    normal_comparison = stats.norm.pdf(prior_example_grid, loc=0, scale=t_scale)
    
    ax_t.plot(prior_example_grid, t_prior, 'darkred', linewidth=3, label=f'Student-t (df={t_df})')
    ax_t.plot(prior_example_grid, normal_comparison, 'gray', linewidth=2, linestyle='--', alpha=0.5, label='Normal (comparison)')
    ax_t.fill_between(prior_example_grid, 0, t_prior, color='red', alpha=0.3)
    
    ax_t.set_xlabel('Parameter Value', fontsize=11)
    ax_t.set_ylabel('Density', fontsize=11)
    ax_t.set_title('Student-t Prior', fontsize=12, fontweight='bold', color='darkred')
    ax_t.set_xlim(-5, 5)
    ax_t.set_ylim(0, None)
    ax_t.legend(fontsize=10)
    ax_t.grid(True, alpha=0.3)
    st.pyplot(fig_t)
    
    st.markdown("""
    **Student-t:**
    - Heavy tails (robust to outliers)
    - More mass in extremes than Normal
    - Degrees of freedom control tail weight
    - Good for uncertain priors
    - Reduces to Normal as df → ∞
    """)

with col_info5:
    # CAUCHY PRIOR (very heavy tails)
    fig_cauchy, ax_cauchy = plt.subplots(figsize=(6, 5))
    
    cauchy_prior = stats.cauchy.pdf(prior_example_grid, loc=0, scale=cauchy_scale)
    ax_cauchy.plot(prior_example_grid, cauchy_prior, 'darkblue', linewidth=3, label=f'Cauchy(0, {cauchy_scale})')
    ax_cauchy.fill_between(prior_example_grid, 0, cauchy_prior, color='blue', alpha=0.3)
    
    ax_cauchy.set_xlabel('Parameter Value', fontsize=11)
    ax_cauchy.set_ylabel('Density', fontsize=11)
    ax_cauchy.set_title('Cauchy Prior', fontsize=12, fontweight='bold', color='darkblue')
    ax_cauchy.set_xlim(-5, 5)
    ax_cauchy.set_ylim(0, 1)
    ax_cauchy.legend(fontsize=10)
    ax_cauchy.grid(True, alpha=0.3)
    st.pyplot(fig_cauchy)
    
    st.markdown("""
    **Cauchy:**
    - Very heavy tails
    - No defined mean or variance!
    - Student-t with df=1
    - Weakly informative
    - Used for hierarchical variance
    """)

with col_info6:
    # LAPLACE PRIOR (sparse/LASSO)
    fig_laplace, ax_laplace = plt.subplots(figsize=(6, 5))
    
    laplace_prior = stats.laplace.pdf(prior_example_grid, loc=0, scale=laplace_scale)
    normal_comparison2 = stats.norm.pdf(prior_example_grid, loc=0, scale=laplace_scale)
    
    ax_laplace.plot(prior_example_grid, laplace_prior, 'teal', linewidth=3, label=f'Laplace(0, {laplace_scale})')
    ax_laplace.plot(prior_example_grid, normal_comparison2, 'gray', linewidth=2, linestyle='--', alpha=0.5, label='Normal (comparison)')
    ax_laplace.fill_between(prior_example_grid, 0, laplace_prior, color='teal', alpha=0.3)
    
    ax_laplace.set_xlabel('Parameter Value', fontsize=11)
    ax_laplace.set_ylabel('Density', fontsize=11)
    ax_laplace.set_title('Laplace Prior', fontsize=12, fontweight='bold', color='teal')
    ax_laplace.set_xlim(-5, 5)
    ax_laplace.set_ylim(0, None)
    ax_laplace.legend(fontsize=10)
    ax_laplace.grid(True, alpha=0.3)
    st.pyplot(fig_laplace)
    
    st.markdown("""
    **Laplace:**
    - Sharp peak at mode
    - Encourages sparsity
    - Equivalent to L1 penalty (LASSO)
    - Used for variable selection
    - More mass at zero than Normal
    """)


# ============================================================================
# SECTION D: MODEL COMPARISON AND BAYES FACTORS
# ============================================================================
st.markdown("<div style='margin-bottom:150px;'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("## Model Comparison with Bayes Factors")

st.markdown("""
**Question:** Is there evidence for a slope effect, or is β = 0?

We compare two hypotheses:
- **H₁ (Alternative):** β ~ N(μ_β, τ_β²) — β can be any value (our Bayesian model)
- **H₀ (Null):** β = β₀ — β is fixed at the null value (e.g., 0)

The **Bayes Factor** quantifies the relative evidence:

**BF₁₀ = p(data | H₁) / p(data | H₀)**

This is the ratio of marginal likelihoods (evidence) for each model.
""")

# Interpret Bayes Factor
def interpret_bf(bf):
    """Interpret Bayes Factor using Kass & Raftery (1995) scale"""
    if bf < 1:
        strength = "Negative"
        direction = "H₀"
        abs_bf = 1/bf
    else:
        direction = "H₁"
        abs_bf = bf
        if abs_bf < 3:
            strength = "Barely worth mentioning"
        elif abs_bf < 10:
            strength = "Substantial"
        elif abs_bf < 30:
            strength = "Strong"
        elif abs_bf < 100:
            strength = "Very strong"
        else:
            strength = "Decisive"
    return strength, direction, abs_bf

strength, direction, abs_bf = interpret_bf(bf_10)

col7, col8 = st.columns([1, 1])

with col7:
    st.markdown(f"""
    <div class="interpretation-box">
    <b>Bayes Factor Results:</b><br><br>
    
    <b>BF₁₀ = {bf_10:.4f}</b><br>
    <b>log(BF₁₀) = {log_bf_10:.4f}</b><br><br>
    
    <b>Interpretation:</b><br>
    The data are <b>{abs_bf:.2f} times more likely</b> under {direction} than under the Null hypothesis.<br><br>
    
    <b>Evidence strength:</b> {strength}<br>
    (Following Kass & Raftery, 1995 interpretation scale)
    </div>
    """, unsafe_allow_html=True)

with col8:
    st.markdown("""
    **Interpreting Bayes Factors:**
    
    | BF₁₀ | Evidence for H₁ |
    |------|-----------------|
    | 1-3 | Anecdotal |
    | 3-10 | Moderate |
    | 10-30 | Strong |
    | 30-100 | Very strong |
    | >100 | Extreme |
    
    **Key advantages:**
    - Quantifies evidence for both hypotheses
    - Can support the null (BF₁₀ < 1)
    - Not affected by optional stopping
    - Incorporates prior information
    """)

# ============================================================================
# SECTION E: KEY TAKEAWAYS
# ============================================================================

st.markdown("---")
st.markdown("## Key Takeaways: Frequentist vs Bayesian")

col9, col10 = st.columns(2)

with col9:
    st.markdown("""
    ### Frequentist Approach
    
    **Strengths:**
    - No need to specify priors
    - Well-understood properties
    - Widely used and accepted
    
    **Limitations:**
    - No probability statements about parameters
    - Confidence intervals often misinterpreted
    - Cannot incorporate prior knowledge
    - Limited model comparison tools
    - Must rely on p-values (binary decisions)
    """)

with col10:
    st.markdown("""
    ### Bayesian Approach
    
    **Strengths:**
    - Direct probability statements about parameters
    - Natural incorporation of prior knowledge
    - Coherent framework for model comparison (Bayes factors)
    - Full posterior distribution (not just point estimates)
    - Handles small samples well via regularization
    
    **Considerations:**
    - Requires specifying priors
    - Computationally intensive for complex models (though not here!)
    - Results depend on prior choice (though this can be explored)
    """)

st.markdown("""
---
### Fundamental Bayesian Concepts Covered:

1. **Prior Distribution:** Our beliefs about parameters before seeing data
2. **Likelihood:** The probability of observing the data given parameters
3. **Posterior Distribution:** Updated beliefs after seeing data (via Bayes' theorem)
4. **Credible Intervals:** Probability statements about where parameters lie
5. **Point Estimates:** Posterior mean, median, or mode
6. **Shrinkage/Regularization:** How prior pulls estimates toward prior mean
7. **Bayes Factors:** Quantifying relative evidence for competing hypotheses
8. **Model Comparison:** Principled way to compare different models

### Try This:
- Change the **sample size** — see how likelihood dominates with more data
- Adjust the **prior means** — see how they influence the posteriors with small n
- Make the **priors very informative** (small τ) vs **uninformative** (large τ)
- Compare **Bayes factors** for different null values
- Try different true intercepts and slopes
""")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
Built for research audiences • Interactive teaching tool • 
Frequentist vs Bayesian comparison with conjugate priors
</div>
""", unsafe_allow_html=True)