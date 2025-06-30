"""
Plots for the emissions plot
"""

from common.dash import dcc, dbc, html, Input, Output, PreventUpdate
from common import data, params

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        html.H4("Regional emissions, with baseline:"),
        dcc.Loading(
            dcc.Graph(id="tabs-regional-emissions-plot", className="fixed_height_plot"),
        ),
        html.H4("Global emissions and temperature path:"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Loading(
                            dcc.Graph(
                                id="tabs-global-emissions-plot",
                                className="fixed_height_plot",
                            )
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        dcc.Loading(
                            dcc.Graph(
                                id="tabs-global-temperature-plot",
                                className="fixed_height_plot",
                            )
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="tabs-plot-container",
)


## Regional emissions
@app.callback(
    Output("tabs-regional-emissions-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_plot_regional_emissions(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["baseline", "regional_emissions"],
        timerange,
        yaxis_title="Emissions",
        height=params.FIXED_PLOT_HEIGHT,
    )

    return fig


## Global emissions and cumulative emissions
@app.callback(
    Output("tabs-global-emissions-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_plot_global_emissions(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["global_emissions", "cumulative_emissions"],
        timerange,
        yaxis_title="Emissions",
        hidden_variables=["cumulative_emissions"],
        colors=[2, 3],
        height=params.FIXED_PLOT_HEIGHT,
    )

    return fig


## Temperature
@app.callback(
    Output("tabs-global-temperature-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_plot_temperature(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["temperature"],
        timerange,
        yaxis_title="GMST",
        colors=[4],
        height=params.FIXED_PLOT_HEIGHT,
    )

    return fig
