"""
Tab for the GDP plots:
 - GDP, capital stock and consumption with absolute vs per capita switch
 - Utility
"""

from common.dash import dcc, dbc, html, Input, Output, PreventUpdate
from common import data

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        html.H4("GDP, capital stock and consumption:"),
        dbc.RadioItems(
            options=[
                {"label": "Absolute", "value": False},
                {"label": "Per capita", "value": True},
            ],
            value=False,
            inline=True,
            id="tabs-GDP-percapita",
        ),
        dcc.Loading(dcc.Graph(id="tabs-GDP-plot")),
        html.H4("Utility:"),
        dcc.Loading(dcc.Graph(id="tabs-GDP-utility-plot")),
    ],
    className="tabs-plot-container",
)


## GDP, capital stock and consumption
@app.callback(
    Output("tabs-GDP-plot", "figure"),
    [
        Input("plot-data-store", "data"),
        Input("plot-timerange", "value"),
        Input("tabs-GDP-percapita", "value"),
    ],
)
def update_gdp_plot(databases, timerange, percapita):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["GDP_gross", "GDP_net", "capital_stock", "consumption"],
        timerange,
        yaxis_title="",
        percapita=percapita,
        hidden_variables=["capital_stock"],
    )
    return fig


## Carbon prices
@app.callback(
    Output("tabs-GDP-utility-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_utility_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(databases, ["utility"], timerange, colors=[4])
    return fig
