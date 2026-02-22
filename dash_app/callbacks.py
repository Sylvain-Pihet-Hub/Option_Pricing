import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from dash import Input, Output, State, html, dcc, no_update

from option_pricing import SecurityData, BinomialModel, MonteCarloModel, BlackScholesModel
from layout_page import slabel, info_row

# Colour tokens
C_CALL   = "#09ab3b"   # green  — call
C_PUT    = "#0068c9"   # blue   — put
C_ACCENT = "#ff4b4b"   # red    — primary / history chart
C_MUTED  = "#808495"   # grey
C_BORDER = "#e6eaf1"   # light border
C_PANEL  = "#ffffff"   # white card
C_INK    = "#262730"   # near-black text

MODEL_LABELS = {
    "bs": "Black-Scholes Model",
    "mc": "Monte Carlo Simulation",
    "binomial": "Binomial Model",
}

def reformat_amount(value):
    return f"${value:,.2f}"

def reformat_percentage(value):
    return f"{value:.2%}"

def days_to_years(exercise_date_str):
    """Convert an ISO date string to T in years from today."""
    exercise = datetime.strptime(exercise_date_str[:10], "%Y-%m-%d").date()
    days = (exercise - datetime.now().date()).days
    return max(days, 1) / 365.0, days

def register_callbacks(app):

    # Model subheader
    @app.callback(
        Output("model-subheader", "children"),
        Input("model-select", "value"),
    )
    def display_model_type(model):
        return f"Pricing Method: {MODEL_LABELS.get(model, model)}"

    # Ticker fetch → spot price, dynamic strike range
    @app.callback(
        Output("ticker-chip","children"),
        Output("stock-store","data"),
        Output("strike-input","value"),
        Output("strike-input","min"),
        Output("strike-input","max"),
        Output("strike-caption","children"),
        Input("ticker-input","value"),
    )
    def fetch_ticker(ticker):
        empty = ("", {}, None, 0.01, 1e9, "")
        if not ticker or not ticker.strip():
            return empty

        try:
            security  = SecurityData(ticker.strip().upper())
            S0 = float(security.current_price)
            returns = security.prices.pct_change().dropna()
            vol = float(returns.std() * np.sqrt(252))

            min_k = round(max(0.1, S0 * 0.75), 2)
            max_k = round(S0 * 1.3, 2)
            def_k = round(S0, 2)

            chip = html.Div([
                html.Div([
                    html.Span(ticker.strip().upper(), className="chip-ticker"),
                    html.Span(f"${S0:,.2f}", className="chip-price"),
                ], className="chip-row"),
                html.Div(f"Annualised volatility: {vol:.1%}", className="chip-vol"),
            ], className="ticker-chip")

            caption = f"Suggested range: ${min_k:,.2f} – ${max_k:,.2f}"
            store = {"ticker": ticker.strip().upper(), "price": S0,
                     "hist_vol": vol, "prices": security.prices.tolist(),
                     "dates": [str(d)[:10] for d in security.prices.index]}

            return chip, store, def_k, min_k, max_k, caption

        except Exception as exc:
            err = html.Div(f"⚠  {exc}", className="err")
            return err, {}, None, 0.01, 1e9, ""

    # Model-specific parameter panel
    @app.callback(
        Output("model-params", "children"),
        Input("model-select",  "value"),
    )
    def fetch_model_params(model):
        hidden = lambda id_, val: dcc.Input(id=id_, type="hidden", value=val)

        if model == "bs":
            # No extra params for Black-Scholes beyond shared inputs
            return html.Div([
                hidden("n-periods-input",252),
                hidden("n-simulations-input",10_000),
                hidden("option-style", None)])

        elif model == "mc":
            return html.Div([
                html.Div([
                    slabel("Number of Simulations",
                           "More simulations increase accuracy but take longer to compute."),
                    dcc.Slider(
                        id="n-simulations-input",
                        min=100, max=100_000, step=100, value=10_000,
                        marks={100: "100", 10_000: "10k", 50_000: "50k", 100_000: "100k"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="slider",
                    ),
                ], className="form-group slider-group"),
                html.Div([
                    slabel("Number of Steps per Path",
                           "Number of time steps simulated along each price path."),
                    dcc.Slider(
                        id="n-periods-input",
                        min=10, max=504, step=1, value=252,
                        marks={10: "10", 63: "Q", 126: "6M", 252: "1Y", 504: "2Y"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="slider",
                    ),
                ], className="form-group slider-group"),
                hidden("option-style", "european"),
            ])

        else:  # binomial
            return html.Div([
                html.Div([
                    slabel("Option Style",
                           "European: exercisable at expiry only. "
                           "American: exercisable at any time before expiry."),
                    dcc.Dropdown(
                        id="option-style",
                        options=[
                            {"label": "European", "value": "european"},
                            {"label": "American", "value": "american"},
                        ],
                        value="european",
                        clearable=False,
                        className="dd",
                    ),
                ], className="form-group"),
                html.Div([
                    slabel("Number of Time Steps",
                           "More steps increase accuracy but take longer to compute."),
                    dcc.Slider(
                        id="n-periods-input",
                        min=50, max=1000, step=50, value=200,
                        marks={50: "50", 200: "200", 500: "500", 1000: "1000"},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="slider",
                    ),
                ], className="form-group slider-group"),
                hidden("n-simulations-input", 10_000),
            ])


    # Main calculation
    @app.callback(
        Output("results-area","children"),
        Output("calc-error","children"),
        Input("calc-btn","n_clicks"),
        State("stock-store","data"),
        State("model-select","value"),
        State("strike-input","value"),
        State("rate-slider","value"),
        State("sigma-slider","value"),
        State("exercise-date","date"),
        State("n-periods-input","value"),
        State("n-simulations-input","value"),
        State("option-style","value"),
        prevent_initial_call=True,
    )
    def calculate_option_price(n_clicks, stock, model, K, rate_pct, sigma_pct, exercise_date, n_periods, n_simulations, option_style):
        # Validation
        if not stock or not stock.get("price"):
            return no_update, "Enter a valid ticker first."
        if not K:
            return no_update, "Strike price K is required."
        if not exercise_date:
            return no_update, "Exercise date is required."

        S0 = float(stock["price"])
        K = float(K)
        r = float(rate_pct)  / 100.0
        sigma = float(sigma_pct) / 100.0
        c = 0.0
        T, days_to_maturity = days_to_years(exercise_date)

        try:
            if model == "bs":
                option_model = BlackScholesModel(S0=S0, T=T, K=K, r=r, c=c, sigma=sigma)
                call_price = option_model.price_call_option()
                put_price  = option_model.price_put_option()
                model_label = "Black-Scholes Model"
                params = [
                    ("Spot Price  S", f"${S0:,.2f}"),
                    ("Strike  K", f"${K:,.2f}"),
                    ("Days to Maturity", str(days_to_maturity)),
                    ("T (years)", f"{T:.6f}"),
                    ("Risk-Free Rate  r", reformat_percentage(r)),
                    ("Volatility  σ", reformat_percentage(sigma)),
                    ("d₁", f"{option_model.d1:.6f}"),
                    ("d₂", f"{option_model.d2:.6f}"),
                ]

            elif model == "mc":
                n_periods = int(n_periods) if n_periods else 252
                n_simulations = int(n_simulations) if n_simulations    else 10_000
                option_model = MonteCarloModel(S0=S0, T=T, n_periods=n_periods, r=r, c=c, sigma=sigma)
                call_price = option_model.price_call_option(K=K)
                put_price  = option_model.price_put_option(K=K)
                # Generate a few paths for the visualisation
                paths = option_model.create_security_price_paths(n_simulations=n_simulations)
                model_label = "Monte Carlo Simulation"
                params = [
                    ("Spot Price S0", f"${S0:,.2f}"),
                    ("Strike K", f"${K:,.2f}"),
                    ("Days to Maturity", str(days_to_maturity)),
                    ("T (years)", f"{T:.6f}"),
                    ("Risk-Free Rate r", reformat_percentage(r)),
                    ("Volatility σ", reformat_percentage(sigma)),
                    ("Simulations", f"{n_simulations:,}"),
                    ("Steps per path", str(n_periods)),
                    ("Random seed", "120"),
                ]

            else:
                n_periods = int(n_periods) if n_periods else 200
                option_type = option_style if option_style else "european"
                option_model = BinomialModel(S0=S0, T=T, n_periods=n_periods, r=r, c=c, sigma=sigma)
                call_lat = option_model.price_call_option(K=K, option_type=option_type)
                put_lat = option_model.price_put_option( K=K, option_type=option_type)
                call_price = call_lat[option_model.n_periods, 0]
                put_price = put_lat[option_model.n_periods, 0]
                model_label = f"Binomial Model {option_type.capitalize()}"
                params = [
                    ("Spot Price S0", f"${S0:,.2f}"),
                    ("Strike K", f"${K:,.2f}"),
                    ("Days to Maturity", str(days_to_maturity)),
                    ("T (years)", f"{T:.6f}"),
                    ("Risk-Free Rate r", reformat_percentage(r)),
                    ("Volatility σ", reformat_percentage(sigma)),
                    ("Option Style", option_type.capitalize()),
                    ("Time Steps N", str(n_periods)),
                    ("Up Factor u", f"{option_model.up:.6f}"),
                    ("Down Factor d", f"{option_model.down:.6f}"),
                    ("Risk-Neutral q", f"{option_model.q:.6f}"),
                    ("Δt", f"{option_model.delta_t:.8f}"),
                ]

        except Exception as exc:
            return no_update, f"{exc}"

        # Historical price chart
        dates = stock.get("dates", [])
        prices = stock.get("prices", [])
        hist_fig = go.Figure()
        if dates and prices:
            hist_fig.add_trace(go.Scatter(
                x=dates, y=prices,
                name="Close Price",
                line=dict(color=C_ACCENT, width=1.8),
                fill="tozeroy",
                fillcolor="rgba(200,114,42,0.07)",
            ))
        hist_fig.update_layout(
            title=dict(text=f"{stock.get('ticker','')}  ·  1Y Price History",
                       font=dict(family="IBM Plex Mono", size=12, color=C_MUTED)),
            paper_bgcolor=C_PANEL, plot_bgcolor="#f0f2f6",
            font=dict(family="IBM Plex Mono", color=C_MUTED, size=11),
            margin=dict(l=16, r=16, t=40, b=16),
            xaxis=dict(gridcolor=C_BORDER, color=C_MUTED, zeroline=False),
            yaxis=dict(title="Price ($)", gridcolor=C_BORDER, color=C_MUTED, zeroline=False),
            showlegend=False,
            hovermode="x unified",
        )

        # Monte Carlo paths chart
        mc_chart = None
        if model == "mc":
            n_show = min(200, n_simulations)
            t_axis = np.linspace(0, T, n_periods + 1)
            mc_fig = go.Figure()
            for i in range(n_show):
                mc_fig.add_trace(go.Scatter(
                    x=t_axis, y=paths[i],
                    mode="lines",
                    line=dict(width=0.6,
                              color=f"rgba(200,114,42,{0.15 if i < n_show else 0})"),
                    showlegend=False,
                    hoverinfo="skip",
                ))
            mc_fig.add_hline(y=K, line_dash="dash", line_color=C_INK,
                             annotation_text=f"K = {K:.2f}",
                             annotation_font_color=C_INK)
            mc_fig.update_layout(
                title=dict(text=f"Monte Carlo {n_show} simulated paths",
                           font=dict(family="IBM Plex Mono", size=12, color=C_MUTED)),
                paper_bgcolor=C_PANEL, plot_bgcolor="#f0f2f6",
                font=dict(family="IBM Plex Mono", color=C_MUTED, size=11),
                margin=dict(l=16, r=16, t=40, b=16),
                xaxis=dict(title="Time (years)", gridcolor=C_BORDER,
                           color=C_MUTED, zeroline=False),
                yaxis=dict(title="Price ($)", gridcolor=C_BORDER,
                           color=C_MUTED, zeroline=False),
                hovermode=False,
            )
            mc_chart = html.Div([
                html.Div("Simulated Price Paths", className="section-hdr"),
                html.Div(
                    dcc.Graph(figure=mc_fig, config={"displayModeBar": False},
                              style={"height": "320px"}),
                    className="chart-card",
                ),
            ])

        # Payoff at expiry chart
        spot_range = np.linspace(S0 * 0.4, S0 * 1.6, 300)
        call_payoff = np.maximum(spot_range - K, 0)
        put_payoff = np.maximum(K - spot_range, 0)

        payoff_fig = go.Figure()
        payoff_fig.add_trace(go.Scatter(
            x=spot_range, y=call_payoff, name="Call Payoff",
            line=dict(color=C_CALL, width=2.5),
            fill="tozeroy", fillcolor="rgba(29,106,74,0.07)",
        ))
        payoff_fig.add_trace(go.Scatter(
            x=spot_range, y=put_payoff, name="Put Payoff",
            line=dict(color=C_PUT, width=2.5),
            fill="tozeroy", fillcolor="rgba(155,35,53,0.07)",
        ))
        payoff_fig.add_vline(x=float(S0), line_dash="dot", line_color=C_MUTED,
                             annotation_text=f"S={S0:.2f}",
                             annotation_font_color=C_MUTED,
                             annotation_position="top right")
        payoff_fig.add_vline(x=float(K), line_dash="dash", line_color=C_INK,
                             annotation_text=f"K={K:.2f}",
                             annotation_font_color=C_INK,
                             annotation_position="top left")
        payoff_fig.update_layout(
            paper_bgcolor=C_PANEL, plot_bgcolor="#f0f2f6",
            font=dict(family="IBM Plex Mono", color=C_MUTED, size=11),
            margin=dict(l=16, r=16, t=28, b=16),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_INK, size=11)),
            xaxis=dict(title="Spot at Expiry", gridcolor=C_BORDER,
                       color=C_MUTED, zeroline=False),
            yaxis=dict(title="Payoff ($)", gridcolor=C_BORDER,
                       color=C_MUTED, zeroline=False),
            hovermode="x unified",
        )

        # Put-call parity check
        parity_lhs = call_price - put_price
        parity_rhs = S0 * np.exp(-c * T) - K * np.exp(-r * T)
        parity_err = abs(parity_lhs - parity_rhs)

        # Assemble results
        results = html.Div([
            html.Hr(className="results-divider"),

            # Price history chart
            html.Div("Price History", className="section-hdr"),
            html.Div(
                dcc.Graph(figure=hist_fig, config={"displayModeBar": False},
                          style={"height": "280px"}),
                className="chart-card",
            ),

            # MC paths (only for Monte Carlo)
            mc_chart,

            # Call / Put result cards
            html.Div([
                html.Div([
                    html.Div("CALL OPTION PRICE", className="res-type call"),
                    html.Div(reformat_amount(call_price), className="res-price call"),
                    html.Div(
                        f"Intrinsic  ${max(S0-K,0):.2f}  ·  "
                        f"Time value  ${max(call_price-max(S0-K,0),0):.2f}",
                        className="res-intrinsic"),
                ], className="res-card call"),
                html.Div([
                    html.Div("PUT OPTION PRICE", className="res-type put"),
                    html.Div(reformat_amount(put_price), className="res-price put"),
                    html.Div(
                        f"Intrinsic ${max(K-S0,0):.2f}  ·  "
                        f"Time value  ${max(put_price-max(K-S0,0),0):.2f}",
                        className="res-intrinsic"),
                ], className="res-card put"),
            ], className="res-grid"),

            # Parameters summary
            html.Div("Parameters", className="section-hdr"),
            html.Div([info_row(k, v) for k, v in params], className="info-table"),

            # Put-call parity
            html.Div("Put-Call Parity Check", className="section-hdr"),
            html.Div([
                info_row("C − P",f"${parity_lhs:.6f}"),
                info_row("S·e^(−cT) − K·e^(−rT)",f"${parity_rhs:.6f}"),
                info_row("Absolute error",f"${parity_err:.2e}"),
            ], className="info-table"),

            # Payoff diagram
            html.Div("Payoff at Expiry", className="section-hdr"),
            html.Div(
                dcc.Graph(figure=payoff_fig, config={"displayModeBar": False},
                          style={"height": "300px"}),
                className="chart-card",
            ),

        ])

        return results, ""