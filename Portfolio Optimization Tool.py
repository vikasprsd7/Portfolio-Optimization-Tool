import numpy as np
import pandas as pd
import torch
import math


## Detect if a GPU is available, otherwise use CPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Step 1: Define your portfolio
assets = ['IBM', 'GOOGLE', 'AMAZON', 'MICROSOFT', 'APPLE', 'SAP', 'SALESFORCE'] 
initial_prices = torch.tensor([175.0, 135.0, 125.0, 310.0, 25.0, 430.0, 350.0], device= device)

# Expected annual returns (approximate)
mu = torch.tensor([0.12, 0.10, 0.15, 0.11, 0.20, 0.08, 0.09], device= device)

# Annual volatilities (approximate)
sigma = torch.tensor([0.30, 0.25, 0.35, 0.28, 0.50, 0.15, 0.18], device= device)

# Custom target allocation ranges (min/max per asset, sum to 1)
min_alloc = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10], device= device)
max_alloc = torch.tensor([0.25, 0.20, 0.20, 0.25, 0.20, 0.30, 0.25], device= device)


num_simulations = 500_000
num_days = 252


# Use the 'device' variable in your tensors
initial_prices = torch.tensor([175.0, 135.0, 125.0, 310.0, 25.0, 430.0, 350.0], device= device)
mu = torch.tensor([0.12, 0.10, 0.15, 0.11, 0.20, 0.08, 0.09], device=device)
# Step 2: Simulate asset RETURNS using GBM instead of just ending prices
dt = 1 / num_days
# We simulate the percentage return for each asset across 500,000 scenarios
asset_returns = torch.zeros((num_simulations, len(assets)), device=device)

for i, (mu_i, sigma_i) in enumerate(zip(mu, sigma)):
    rand = torch.randn(num_simulations, num_days, device=device)
    # Total accumulated return over 252 days for this scenario
    total_log_return = ((mu_i - 0.5 * sigma_i**2) * dt * num_days) + (sigma_i * math.sqrt(dt) * rand.sum(dim=1))
    asset_returns[:, i] = torch.exp(total_log_return) - 1

# Step 3: Generate random portfolio weights within min/max bounds
weights = torch.rand((num_simulations, len(assets)), device=device)
weights = min_alloc + weights * (max_alloc - min_alloc)
weights /= weights.sum(dim=1, keepdim=True)

# Calculate individual expected returns for all 500,000 portfolios
# (500k weights multiplied by the mean simulated return of each asset)
mean_asset_returns = asset_returns.mean(dim=0) # shape: (7,)
portfolio_expected_returns = (weights * mean_asset_returns).sum(dim=1) # shape: (500000,)

# Calculate the actual variance/covariance of our simulated asset returns
# We need this to calculate the unique volatility of EACH random weight combination
centered_returns = asset_returns - mean_asset_returns
covariance_matrix = (centered_returns.T @ centered_returns) / (num_simulations - 1)

# Formula for portfolio variance: w^T * Sigma * w
# We compute this efficiently across all 500,000 portfolios using batch matrix multiplication
portfolio_variance = torch.bmm(weights.unsqueeze(1), covariance_matrix.expand(num_simulations, -1, -1))
portfolio_variance = torch.bmm(portfolio_variance, weights.unsqueeze(2)).squeeze()
portfolio_individual_volatility = torch.sqrt(portfolio_variance)

# Sharpe ratio assuming risk-free rate = 0 (Calculated per individual portfolio!)
sharpe_ratio = portfolio_expected_returns / portfolio_individual_volatility

# Find best portfolio
best_idx = torch.argmax(sharpe_ratio)
best_weights = weights[best_idx].cpu().numpy()

# Step 4: Output results
print("\nOptimal Portfolio Allocation (based on Sharpe Ratio):")
for asset, weight in zip(assets, best_weights):
    print(f"{asset}: {weight:.2%}")

print(f"\nExpected Portfolio Return: {portfolio_expected_returns[best_idx].item():.2%}")
print(f"Portfolio Volatility: {portfolio_individual_volatility[best_idx].item():.2%}")
print(f"Sharpe Ratio: {sharpe_ratio[best_idx].item():.2f}")

# Save all simulated portfolio data
pd.DataFrame({
    'Return': portfolio_expected_returns.cpu().numpy(),
    'Volatility': portfolio_individual_volatility.cpu().numpy(),
    'Sharpe': sharpe_ratio.cpu().numpy()
}).to_csv('gpu_portfolio_simulation.csv', index=False)

print("\nSimulation complete! CSV saved with genuine optimized portfolios.")
