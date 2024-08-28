"""
Provides a table with the differences between the selected scenarios.
"""

import pandas as pd
from app import app
from common.dash import dbc, dcc, Input, Output, html


layout = dbc.Row(
    [
        dbc.Col(html.P("Scenarios:"), md=1),
        dbc.Col(html.Div(id="legend-table")),
        dbc.Col([], md=2),
    ]
)


@app.callback(
    Output("legend-table", "children"),
    [Input("plot-data-store", "data")],
)
def update_legend(databases):
    if (
        databases is None or len(databases) == 0
    ):  # Only update when multiple files are selected
        return []
    print(databases.keys())

    differences = param_differences(databases)

    headers = [
        html.Th(create_empty_legend(name, databases[name]["meta"]["line_dash"]))
        for name in differences.columns
    ]
    table_header = [html.Thead(html.Tr([html.Td()] + headers))]

    rows = [
        html.Tr([html.Th(param)] + [html.Td(str(value)) for value in row])
        for param, row in differences.iterrows()
    ]

    table_body = [html.Tbody(rows)]

    return dbc.Table(table_header + table_body, size="sm", hover=True)


def create_empty_legend(name, dash):
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": [None],
                "y": [None],
                "mode": "lines",
                "name": f"<b>{name}</b>",
                "line": {
                    "color": "black",
                    "dash": dash,
                },
                "showlegend": True,
            }
        ],
        "layout": {
            "height": 30,
            "width": 300,
            "margin": {"t": 0, "l": 0, "r": 0, "b": 0},
            "legend": {
                "orientation": "h",
                "y": 0.5,
                "yanchor": "middle",
                "font": {"size": 13},
            },
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {
                "showgrid": False,
                "zeroline": False,
                "visible": False,
            },
            "yaxis": {
                "showgrid": False,
                "zeroline": False,
                "visible": False,
            },
        },
    }

    return dcc.Graph(figure=fig)


def param_differences(all_params) -> pd.DataFrame:
    # First, flatten the nested parameter dictionaries
    flattened = {
        name: flatten(single_params) for name, single_params in all_params.items()
    }

    differences = {}

    # Loop through each flattened dict to find differences
    all_keys = [set(x.keys()) for x in flattened.values() if len(x) > 0]

    if len(all_keys) <= 1:
        return pd.DataFrame(columns=list(all_params.keys()))

    keys_union = set.union(*all_keys)
    keys_intersect = set.intersection(*all_keys)
    keys_difference = keys_union - keys_intersect

    for key in keys_intersect:
        # Check if all values are equal
        unique_values = set(
            params[key] for params in flattened.values() if len(params) > 0
        )
        if len(unique_values) > 1:
            keys_difference.add(key)

    # Loop through each key_difference to get the respective values
    differences = {
        key: {name: params.get(key, "(missing)") for name, params in flattened.items()}
        for key in keys_difference
    }

    if len(differences) == 0:
        return pd.DataFrame(columns=list(all_params.keys()))

    return pd.DataFrame(differences).T.sort_index()


def recursive_traverse(name, subset, flattened):
    for key, value in subset.items():
        if key not in ["regions", "regionstype"]:  # Regions are checked separately
            newname = f"{name} - {key}" if name != "" else key
            if type(value) == dict:
                recursive_traverse(newname, value, flattened)
            else:
                # Check if newname is one of the constraint_variables:
                if " - constraint_variables - " in newname:
                    newname_split = newname.split(" - ")
                    simpler_newname = " - ".join(newname_split[:3])
                    simpler_value = "(present)"
                    flattened[simpler_newname] = simpler_value
                else:
                    flattened[newname] = str(value)


def flatten(dictionary):
    flattened = {}
    if "params" in dictionary["meta"]:
        recursive_traverse("", dictionary["meta"]["params"], flattened)
    return flattened
