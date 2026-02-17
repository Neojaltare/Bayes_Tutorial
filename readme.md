# Frequentist vs Bayesian Regression Teaching App

An interactive Streamlit application for teaching fundamental Bayesian statistics concepts to research audiences.

## Features

### Core Statistical Concepts Covered:
1. **Prior Distribution** - Initial beliefs about parameters
2. **Likelihood** - Probability of data given parameters
3. **Posterior Distribution** - Updated beliefs after observing data
4. **Bayes' Theorem** - How prior and likelihood combine
5. **Credible Intervals** - Bayesian confidence intervals with probability interpretation
6. **Shrinkage/Regularization** - How priors regularize estimates
7. **Bayes Factors** - Quantifying evidence for competing hypotheses
8. **Model Comparison** - Principled comparison of different models

### Interactive Features:
- Adjust sample size to see how data dominates prior with large n
- Modify prior parameters (mean and variance) to see their influence
- Compare frequentist confidence intervals vs Bayesian credible intervals
- Visualize prior, likelihood, and posterior distributions
- Calculate Bayes factors for model comparison
- Make direct probability statements about parameters

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use for Presentations

### Section Flow:

1. **Section A - Frequentist Approach**
   - Shows point estimates and confidence intervals
   - Highlights what frequentist inference can and cannot tell us
   - Emphasizes that CIs are NOT probability statements about parameters

2. **Section B - Why Bayesian?**
   - Motivates the need for uncertainty quantification
   - Explains the concept of treating parameters as random variables
   - Introduces belief updating

3. **Section C - Bayesian Approach**
   - Shows prior, likelihood, and posterior distributions
   - Demonstrates how beliefs update with data
   - Makes direct probability statements about parameters
   - Compares credible intervals vs confidence intervals

4. **Section D - Model Comparison**
   - Introduces Bayes factors
   - Compares alternative hypothesis vs null hypothesis
   - Quantifies strength of evidence


## License
Free to use for educational purposes.