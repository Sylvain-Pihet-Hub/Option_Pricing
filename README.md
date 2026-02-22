# Option Pricing Models

A quantitative finance web application for pricing European and American options using three industry-standard models: **Black-Scholes**, **Monte Carlo simulation**, and the **Binomial Tree**. Stock prices are fetched in real time from Yahoo Finance.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Backend — option\_pricing.py](#backend--option_pricingpy)
- [Pricing Models](#pricing-models)
  - [Black-Scholes](#1-black-scholes-model)
  - [Monte Carlo](#2-monte-carlo-simulation)
  - [Binomial Tree](#3-binomial-tree-model)
- [Parameter Comparison](#parameter-comparison)
- [Web Application — Dash](#web-application--dash)
- [How to Run](#how-to-run)

---

## Project Structure

```
option-pricing-models/
│
├── README.md
├── requirements.txt
├── option_pricing.py          ← shared backend (models + data)
│
└── dash_app/
    ├── app.py                 ← entry point
    ├── layout.py              ← page structure and components
    ├── callbacks.py           ← interactivity and model dispatch
    └── assets/
        └── style.css          ← custom styling
```

---

## Backend — option_pricing.py

The backend is a pure Python module with no dependency on any web framework. It is responsible for two things:

**Data retrieval.** The `SecurityData` class wraps the `yfinance` library to fetch historical closing prices for any ticker symbol. It exposes the current spot price and the full 1-year price series, which the frontend uses both for live pricing and to plot the historical chart.

**Option pricing.** Three model classes implement the pricing logic independently of each other. Each class accepts the relevant financial parameters at construction time, and exposes two methods: `price_call_option()` and `price_put_option()`. This clean interface means the web layer never needs to know the internal mechanics of any model — it simply instantiates the right class and calls the same two methods regardless of which model the user selected.

---

## Pricing Models

### 1. Black-Scholes Model

The Black-Scholes model provides a closed-form analytical solution for pricing European options. It assumes the underlying asset follows a geometric Brownian motion with constant volatility and drift, and that markets are frictionless with no arbitrage opportunities.

The call price is derived from:

$$C = S_0 e^{-cT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S_0/K) + (r - c + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

The put price is obtained via put-call parity. Because the solution is closed-form, this model is the fastest of the three.

**Limitation:** Only valid for European-style options. Cannot price American options due to the possibility of early exercise.

---

### 2. Monte Carlo Simulation

The Monte Carlo model estimates option prices by simulating a large number of possible price paths for the underlying asset, each driven by random draws from a standard normal distribution. The option payoff is computed at expiry for each path, and the final price is the average discounted payoff across all simulations.

Each price path follows:

$$S_{t+\Delta t} = S_t \cdot \exp\!\left[\left(r - c - \frac{\sigma^2}{2}\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right], \quad Z \sim \mathcal{N}(0,1)$$

The more simulations used, the more accurate the estimate — at the cost of computation time. A fixed random seed (120) ensures reproducible results.

**Limitation:** Prices European options only. Accuracy depends on the number of simulations; very few simulations produce noisy estimates.

---

### 3. Binomial Tree Model

The Binomial Tree model discretises time into `N` equal steps, building a recombining lattice of possible asset prices. At each node, the asset price can move up by factor `u` or down by factor `d = 1/u`, where `u` is derived from the volatility using the Cox-Ross-Rubinstein (CRR) convention:

$$u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad q = \frac{e^{(r-c)\Delta t} - d}{u - d}$$

The option value is computed by backward induction from the terminal payoffs to the root of the tree. This model supports both **European** and **American** options — for American options, at each node the holder's decision to exercise early is compared against the continuation value, and the maximum is taken.

**Advantage over Black-Scholes:** Can price American-style options, which Black-Scholes and Monte Carlo cannot.

---

## Parameter Comparison

| Parameter | Black-Scholes | Monte Carlo | Binomial Tree |
|---|:---:|:---:|:---:|
| Spot price `S` (auto-fetched) | ✓ | ✓ | ✓ |
| Strike price `K` | ✓ | ✓ | ✓ |
| Exercise date → `T` (years) | ✓ | ✓ | ✓ |
| Risk-free rate `r` | ✓ | ✓ | ✓ |
| Volatility `σ` | ✓ | ✓ | ✓ (used to derive `u`) |
| Number of time steps `N` | — | ✓ (steps per path) | ✓ (tree depth) |
| Number of simulations | — | ✓ | — |
| Option style (European / American) | European only | European only | ✓ Both |

> **Note on `T`:** Maturity is entered as an exercise date in the UI. The application converts it to a fraction of a year by computing the number of calendar days to expiry divided by 365.

---

## Web Application — Dash

The frontend is built with [Dash](https://dash.plotly.com/), a Python framework for analytical web applications built on top of Flask and React. The application is split into three files to separate concerns clearly.

**`app.py`** is the entry point. It creates the Dash server instance and wires the layout and callbacks together. Running this file starts the Flask development server.

**`layout.py`** defines the entire visual structure of the page as a tree of Dash HTML and core components. It contains no business logic — only the arrangement of inputs, labels, sliders, and output containers. The sidebar holds the model selector; the main content area holds all shared parameters, model-specific parameters, and the results area.

**`callbacks.py`** contains all the interactivity. Dash callbacks are Python functions decorated with `@app.callback` that fire automatically when an input component changes. Three callbacks handle: (1) fetching the live spot price when the ticker changes and updating the suggested strike range, (2) swapping the model-specific parameter panel when a different model is selected, and (3) running the pricing calculation when the button is clicked and rendering the results — including a historical price chart, call/put price cards, a parameter summary table, put-call parity verification, and a payoff-at-expiry diagram.

The backend `option_pricing.py` is imported directly into `callbacks.py`. The web layer simply maps user inputs to constructor arguments and calls `price_call_option()` / `price_put_option()`.

```
User input → Dash callback (callbacks.py)
                 ↓
           option_pricing.py  (BlackScholesModel / MonteCarloModel / BinomialModel)
                 ↓
           Call & Put prices → rendered to the page
```

---

## How to Run

### Prerequisites

- Python 3.9 or higher
- An internet connection (required to fetch live prices from Yahoo Finance)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/option-pricing-models.git
cd option-pricing-models
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dash application

```bash
cd dash_app
python app.py
```

Then open your browser at:

```
http://127.0.0.1:8050
```

### Dependencies

```
dash>=2.7.0
dash-bootstrap-components>=1.5.0
plotly>=5.18.0
yfinance>=0.2.36
numpy>=1.24.0
scipy>=1.11.0
pandas>=2.0.0
```

---

## Usage

1. **Select a model** from the sidebar (Black-Scholes, Monte Carlo, or Binomial Tree)
2. **Enter a ticker symbol** — the current spot price is fetched automatically and the suggested strike range is updated
3. **Set the parameters** — strike price, exercise date, risk-free rate, and volatility via the sliders; model-specific parameters (number of simulations, time steps, option style) appear automatically
4. **Click Calculate** — results display below the form, including the call and put prices, a 1-year price history chart, model parameters summary, put-call parity check, and payoff-at-expiry diagram