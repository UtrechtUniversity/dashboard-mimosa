"""
Plot any available variable
"""

import pandas as pd
from common.dash import dcc, dbc, html, Input, Output, PreventUpdate
from common import data, params

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        html.H4("Select variable:"),
        dcc.Dropdown(
            id="tabs-all-variables-dropdown1",
            options=[],
            multi=True,
        ),
        dbc.RadioItems(
            options=[
                {"label": "Absolute", "value": False},
                {"label": "Per capita", "value": True},
            ],
            value=False,
            inline=True,
            id="tabs-all-variables-percapita1",
        ),
        dcc.Loading(
            dcc.Graph(id="tabs-all-variables-plot1", className="fixed_height_plot")
        ),
    ],
    className="tabs-plot-container",
)


## List of variables
@app.callback(
    Output("tabs-all-variables-dropdown1", "options"),
    [Input("plot-data-store", "data")],
)
def update_available_variables(databases):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    variables = set()
    for df_i, df_info in enumerate(databases.values()):
        database = df_info["data"]
        if isinstance(database, dict):
            # t0 = time.time()
            database = pd.DataFrame(database)
            # t1 = time.time()
            # print("Took {} seconds".format(t1 - t0))

        variables = variables.union(set(database["Variable"].unique()))

    return [{"label": var, "value": var} for var in sorted(variables)]


## Plot with the selected variables
@app.callback(
    Output("tabs-all-variables-plot1", "figure"),
    [
        Input("plot-data-store", "data"),
        Input("plot-timerange", "value"),
        Input("tabs-all-variables-dropdown1", "value"),
        Input("tabs-all-variables-percapita1", "value"),
    ],
)
def update_plot1(databases, timerange, selected_variables, percapita):
    if (
        databases is None
        or len(databases) == 0
        or selected_variables is None
        or len(selected_variables) == 0
    ):
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        selected_variables,
        timerange,
        yaxis_title="Value",
        height=params.FIXED_PLOT_HEIGHT,
        percapita=percapita,
    )

    return fig
