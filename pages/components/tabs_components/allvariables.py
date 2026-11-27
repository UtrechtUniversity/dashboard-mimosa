"""
Plot any available variable, in one or more separate graphs
"""

import pandas as pd
from common.dash import (
    dcc,
    dbc,
    dash,
    html,
    ALL,
    MATCH,
    Input,
    Output,
    State,
    PreventUpdate,
)
from common import params

from pages.components.tabs_components import plotutils

from app import app


def create_panel(index):
    return html.Div(
        [
            html.Div(
                [
                    html.H4("Select variable:"),
                    html.Button(
                        id={"type": "tabs-all-variables-delete", "index": index},
                        n_clicks=0,
                        className="btn-close",
                        title="Remove graph",
                        style={"fontSize": "0.7rem", "padding": "0.25rem"},
                        **{"aria-label": "Remove graph"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between"},
            ),
            dcc.Dropdown(
                id={"type": "tabs-all-variables-dropdown", "index": index},
                options=[],
                multi=True,
            ),
            dbc.Switch(
                id={"type": "tabs-all-variables-percapita", "index": index},
                label="Divide by population",
                value=False,
            ),
            dcc.Loading(
                dcc.Graph(
                    id={"type": "tabs-all-variables-plot", "index": index},
                    className="fixed_height_plot",
                )
            ),
        ],
        id={"type": "tabs-all-variables-panel", "index": index},
    )


layout = html.Div(
    [
        html.Br(),
        dcc.Store(id="tabs-all-variables-options"),
        html.Div([create_panel(0)], id="tabs-all-variables-container"),
        html.Div(
            dbc.Button(
                "Add another graph",
                id="tabs-all-variables-add",
                n_clicks=0,
                color="secondary",
                outline=True,
                className="mt-2",
            ),
            style={"display": "flex", "justifyContent": "center"},
        ),
    ],
    className="tabs-plot-container",
)


## Add or remove graphs
@app.callback(
    Output("tabs-all-variables-container", "children"),
    Input("tabs-all-variables-add", "n_clicks"),
    Input({"type": "tabs-all-variables-delete", "index": ALL}, "n_clicks"),
    State({"type": "tabs-all-variables-panel", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def update_panels(add_clicks, _delete_clicks, panel_ids):
    # Ignore callbacks caused by delete buttons appearing or disappearing.
    if not dash.ctx.triggered[0]["value"]:
        raise PreventUpdate

    triggered_id = dash.ctx.triggered_id
    children = dash.Patch()

    if triggered_id == "tabs-all-variables-add":
        children.append(create_panel(add_clicks))
        return children

    for position, panel_id in enumerate(panel_ids):
        if panel_id["index"] == triggered_id["index"]:
            del children[position]
            return children
    raise PreventUpdate


## List of variables
@app.callback(
    Output("tabs-all-variables-options", "data"),
    Input("plot-selected-store", "data"),
)
def update_available_variables(databases):
    variables = set()
    for df_info in (databases or {}).values():
        database = df_info["data"]
        if isinstance(database, dict):
            database = pd.DataFrame(database)
        variables.update(database["Variable"].unique())

    return [{"label": var, "value": var} for var in sorted(variables)]


@app.callback(
    Output({"type": "tabs-all-variables-dropdown", "index": MATCH}, "options"),
    Input("tabs-all-variables-options", "data"),
)
def set_dropdown_options(options):
    return options or []


## Plot with the selected variables, separately for each graph
@app.callback(
    Output({"type": "tabs-all-variables-plot", "index": MATCH}, "figure"),
    [
        Input("plot-selected-store", "data"),
        Input("plot-timerange", "value"),
        Input({"type": "tabs-all-variables-dropdown", "index": MATCH}, "value"),
        Input({"type": "tabs-all-variables-percapita", "index": MATCH}, "value"),
    ],
)
def update_plot(databases, timerange, selected_variables, percapita):
    if databases is None or len(databases) == 0:
        return {}
    if selected_variables is None or len(selected_variables) == 0:
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
