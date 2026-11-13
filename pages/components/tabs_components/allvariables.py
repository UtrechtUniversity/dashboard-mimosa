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
from common import data, params

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        # List of indices of the currently shown graphs
        dcc.Store(id="tabs-all-variables-panels", data=[0]),
        html.Div(id="tabs-all-variables-container"),
        html.Div(
            dbc.Button(
                "Add another graph",
                id="tabs-all-variables-add",
                color="secondary",
                outline=True,
                className="mt-2",
            ),
            style={"display": "flex", "justifyContent": "center"},
        ),
    ],
    className="tabs-plot-container",
)


def create_panel(index, selected_variables=None, options=None, percapita=False, figure=None):
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
                options=options or [],
                value=selected_variables,
                multi=True,
            ),
            dbc.RadioItems(
                options=[
                    {"label": "Absolute", "value": False},
                    {"label": "Per capita", "value": True},
                ],
                value=percapita,
                inline=True,
                id={"type": "tabs-all-variables-percapita", "index": index},
            ),
            dcc.Loading(
                dcc.Graph(
                    id={"type": "tabs-all-variables-plot", "index": index},
                    figure=figure,
                    className="fixed_height_plot",
                )
            ),
        ],
    )


## Render one panel per graph in the store
@app.callback(
    Output("tabs-all-variables-container", "children"),
    Input("tabs-all-variables-panels", "data"),
    State({"type": "tabs-all-variables-dropdown", "index": ALL}, "value"),
    State({"type": "tabs-all-variables-dropdown", "index": ALL}, "options"),
    State({"type": "tabs-all-variables-dropdown", "index": ALL}, "id"),
    State({"type": "tabs-all-variables-percapita", "index": ALL}, "value"),
    State({"type": "tabs-all-variables-percapita", "index": ALL}, "id"),
    State({"type": "tabs-all-variables-plot", "index": ALL}, "figure"),
    State({"type": "tabs-all-variables-plot", "index": ALL}, "id"),
)
def render_panels(
    panel_indices,
    dropdown_values,
    dropdown_options,
    dropdown_ids,
    percapitas,
    percapita_ids,
    figures,
    figure_ids,
):
    if not panel_indices:
        return []

    keyed = lambda ids, values: {
        _id["index"]: value for _id, value in zip(ids or [], values or [])
    }

    # Keep the currently selected variables, options and figures when
    # the panels are rebuilt (e.g. when another graph is added)
    selected_variables = keyed(dropdown_ids, dropdown_values)
    options = keyed(dropdown_ids, dropdown_options)
    percapita = keyed(percapita_ids, percapitas)
    figs = keyed(figure_ids, figures)

    return [
        create_panel(
            index,
            selected_variables.get(index),
            options.get(index),
            percapita.get(index),
            figs.get(index),
        )
        for index in panel_indices
    ]


## Add or remove graphs
@app.callback(
    Output("tabs-all-variables-panels", "data"),
    Input("tabs-all-variables-add", "n_clicks"),
    Input({"type": "tabs-all-variables-delete", "index": ALL}, "n_clicks"),
    State("tabs-all-variables-panels", "data"),
    prevent_initial_call=True,
)
def update_panel_list(add_clicks, delete_clicks, panel_indices):
    panel_indices = panel_indices or []
    triggered_id = dash.ctx.triggered_id

    if triggered_id == "tabs-all-variables-add":
        new_index = max(panel_indices) + 1 if panel_indices else 0
        return panel_indices + [new_index]

    # A delete button was (re)registered when the panels were rebuilt: ignore
    if not any(delete_clicks or []):
        raise PreventUpdate

    return [index for index in panel_indices if index != triggered_id["index"]]


## List of variables
@app.callback(
    Output({"type": "tabs-all-variables-dropdown", "index": ALL}, "options"),
    [Input("plot-selected-store", "data")],
)
def update_available_variables(databases):
    if databases is None or len(databases) == 0:
        # Keep the options of every panel empty
        return [[] for _ in dash.ctx.outputs_list]

    variables = set()
    for df_i, df_info in enumerate(databases.values()):
        database = df_info["data"]
        if isinstance(database, dict):
            # t0 = time.time()
            database = pd.DataFrame(database)
            # t1 = time.time()
            # print("Took {} seconds".format(t1 - t0))

        variables = variables.union(set(database["Variable"].unique()))

    options = [{"label": var, "value": var} for var in sorted(variables)]
    return [options for _ in dash.ctx.outputs_list]


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
