from datetime import datetime, timedelta
from dash import dcc, html


def slabel(text, caption=None):
    """Section label with optional caption underneath."""
    children = [html.Div(text, className="slabel")]
    if caption:
        children.append(html.Div(caption, className="caption"))
    return html.Div(children)

def info_row(key, val):
    return html.Div([
        html.Span(key, className="info-key"),
        html.Span(val, className="info-val"),
    ], className="info-row")

# Main layout factory
def create_layout():

    default_exercise = (datetime.today() + timedelta(days=365)).strftime("%Y-%m-%d")
    min_exercise     = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    return html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Span("Ω", style={"color": "#F5F2EC"}),
                html.Span("Pricer", style={"color": "#C8722A"}),
            ], className="hdr-logo"),
            html.Div("Option Pricing Engine", className="hdr-tag"),
        ], className="hdr"),

        # ── Main grid ───────────────────────────────────────────────────────
        html.Div([
            html.Div([

                slabel("Pricing Method"),
                dcc.RadioItems(
                    id="model-select",
                    options=[
                        {"label": "Black-Scholes Model",    "value": "bs"},
                        {"label": "Monte Carlo Simulation", "value": "mc"},
                        {"label": "Binomial Model",         "value": "binomial"},
                    ],
                    value="bs",
                    className="radio-group",
                    labelClassName="radio-label",
                    inputClassName="radio-input",
                ),

            ], className="sidebar"),

            html.Div([

                # Dynamic subheader (e.g. "Pricing method: Black-Scholes Model")
                html.Div(id="model-subheader", className="model-subheader"),

                # ── Shared parameter form ────────────────────────────────
                html.Div([

                    # Ticker
                    html.Div([
                        slabel("Ticker Symbol",
                               "Enter the stock symbol (e.g., AAPL for Apple Inc.)"),
                        dcc.Input(
                            id="ticker-input",
                            type="text",
                            placeholder="e.g. AAPL, TSLA, SPY",
                            value="AAPL",
                            className="finput",
                            debounce=True,
                        ),
                        html.Div(id="ticker-chip"),
                    ], className="form-group"),

                    # Strike price — min/max/default updated by callback
                    html.Div([
                        slabel("Strike Price",
                               "The price at which the option can be exercised."),
                        dcc.Input(
                            id="strike-input",
                            type="number",
                            placeholder="Fetching price…",
                            min=0.01, step=0.01,
                            className="finput",
                            debounce=True,
                        ),
                        html.Div(id="strike-caption", className="caption"),
                    ], className="form-group"),

                    # Risk-free rate slider
                    html.Div([
                        slabel("Risk-Free Rate (%)",
                               "The theoretical rate of return of a risk-free investment. "
                               "Usually based on government bonds."),
                        dcc.Slider(
                            id="rate-slider",
                            min=0, max=20, step=0.1, value=5,
                            marks={0: "0%", 5: "5%", 10: "10%", 15: "15%", 20: "20%"},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="slider",
                        ),
                    ], className="form-group slider-group"),

                    # Volatility slider
                    html.Div([
                        slabel("Sigma — Volatility (%)",
                               "A measure of the stock's price variability. "
                               "Higher values indicate more volatile stocks."),
                        dcc.Slider(
                            id="sigma-slider",
                            min=1, max=100, step=1, value=20,
                            marks={1: "1%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="slider",
                        ),
                    ], className="form-group slider-group"),

                    # Exercise date
                    html.Div([
                        slabel("Exercise Date",
                               "The date when the option can be exercised."),
                        dcc.DatePickerSingle(
                            id="exercise-date",
                            min_date_allowed=min_exercise,
                            date=default_exercise,
                            display_format="YYYY-MM-DD",
                            className="date-picker",
                        ),
                    ], className="form-group"),

                    # Model-specific params injected here by callback
                    html.Div(id="model-params"),

                    # Calculate button + error message
                    html.Div([
                        html.Button(
                            id="calc-btn",
                            children="Calculate Option Price",
                            className="calc-btn",
                            n_clicks=0,
                        ),
                        html.Div(id="calc-error", className="err"),
                    ], className="form-group"),

                ], className="param-form"),

                # ── Results injected here ────────────────────────────────
                html.Div(id="results-area"),

            ], className="content"),

        ], className="grid"),

        # Client-side data store
        dcc.Store(id="stock-store"),
    ])