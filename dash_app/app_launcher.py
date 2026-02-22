import dash
import dash_bootstrap_components as dbc
from layout_page import create_layout
from dash_app.callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Option-Pricer",
    suppress_callback_exceptions=True,
)

# expose Flask server for deployment
server = app.server

app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True, port=8050)