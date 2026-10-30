"""
The input parameters of each file are read using the DataStore.
This is a simple dictionary, which is formatted using pprint.
"""

import pprint

from common.dash import html, Input, Output, PreventUpdate
from common import data

from app import app

layout = html.Div(
    [html.Br(), html.H4("Input parameters:"), html.Pre(id="tabs-inputparams-pre")],
    className="tabs-plot-container",
)


@app.callback(
    Output("tabs-inputparams-pre", "children"), [Input("plot-selected-store", "data")]
)
def update_params(databases):
    if not databases:
        return []
    html_elements = []
    for filename, database in databases.items():
        if "params" in database["meta"]:
            html_elements.extend(
                [
                    html.H5(filename),
                    html.Pre(pprint.pformat(database["meta"]["params"])),
                ]
            )
    return html_elements
