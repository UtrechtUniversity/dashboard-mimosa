"""
Provides the file selection dropdown and time range slider.

When multiple files are selected, this also shows a legend with
the line dash style for each file.
"""

import numpy as np
import pandas as pd

from app import app
from common import data
from common.dash import dbc, dcc, Input, Output, html, PreventUpdate
from pages.components import fileupload


layout_rangeslider = dcc.RangeSlider(
    id="plot-timerange",
    min=2020,
    max=2100,
    step=10,
    value=[2020, 2100],
    marks={int(x): str(int(x)) for x in np.arange(2020, 2200, 10)},
    allow_direct_input=False,
)
layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col([html.P("Experiment:")], md=1),
                dbc.Col([fileupload.layout]),
            ]
        ),
        dbc.Row(
            [
                dbc.Col([html.P("Time range:")], md=1),
                dbc.Col([layout_rangeslider]),
                dbc.Col([], md=2),
            ],
            style={"padding-top": "10px"},
        ),
        dcc.Store(id="plot-selected-store"),
        dcc.Store(id="plot-data-store"),
    ]
)


@app.callback(Output("plot-timerange", "max"), [Input("plot-data-store", "data")])
def update_range(databases):
    if databases is None or len(databases) == 0:
        raise PreventUpdate
    return min(
        [
            max(
                pd.DataFrame(single_df["data"])
                .drop(columns=["Variable", "Region", "Unit"], errors="ignore")
                .columns.to_numpy(dtype="float")
            )
            for single_df in databases.values()
        ]
    )
