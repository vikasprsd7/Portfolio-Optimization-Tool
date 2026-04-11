## Introduction ##


This project implements a **Multi-Asset Portfolio Optimizer** using **Geometric Brownian Motion (GBM)** and **Monte Carlo Simulations**. By simulating 500,000 potential market paths and testing random weight distributions within strict allocation constraints, the tool identifies the **"Maximum Sharpe Ratio"** portfolio to maximize risk-adjusted returns for a diversified FinTech investment strategy.

## Core Logic & How the Code Works ##
The script follows a rigorous **quantitative workflow** to determine the best way to **distribute capital across seven assets**.

- **Geometric Brownian Motion (GBM)** : 
    The "heart" of the simulation is the GBM formula.
  It assumes that stock prices follow a random walk with a consistent trend (drift) and random noise (volatility).

  $$S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)$$
  
  Instead of just looking at historical averages, the code simulates 252 trading days for each of the 500,000 scenarios to predict possible ending prices.

- **Constrained Monte Carlo Simulation** : The code generates random weights but applies min/max constraints (e.g., ensuring at least 10% is in SPY). This is a realistic "FinTech" approach, as real-world funds have diversification requirements to prevent putting all money into a single high-risk asset like NEB.

  
- **Sharpe Ratio Optimization** :  The "Main Logic" is to find the portfolio where the ratio of return to volatility is highest. Since the risk-free rate is set to zero, it simply seeks the highest return per unit of standard deviation.


## Tech Stack & Libraries ##

**Primary Libraries**

- **PyTorch (torch)**:

While typically used for AI, it was used here for Tensor Computing. PyTorch allows us to run 500,000 simulations simultaneously using vectorized operations, which is significantly faster than standard Python loops.

It handles all matrix math, random number generation, and GPU acceleration (via the device variable).

****Note**:In this documentation, I have emphasized PyTorch over NumPy for its speed and ability to handle large-scale simulations (Monte Carlo).**

- **Math**:

Used for basic scalar operations like math.sqrt within the GBM calculation.

