"""
Provides the file selection dropdown and time range slider.

When multiple files are selected, this also shows a legend with
the line dash style for each file.
"""

import base64
import datetime
import io
import json
import pandas as pd

from app import app
from common import data
from common.dash import dbc, dash, dcc, Input, Output, State, html, PreventUpdate


def all_experiments_options():
    options = []
    for filename in data.get_all_experiments():
        name = data.filename_stem(filename)
        options.append({"label": name, "value": filename})
    return options


upload_style = {
    "width": "100%",
    "height": "70px",
    "borderWidth": "1px",
    "borderStyle": "dashed",
    "borderRadius": "5px",
    "textAlign": "center",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "margin": "10px 0",
}

layout_upload = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div(
                            [
                                html.Strong("Replace scenarios"),
                                html.Br(),
                                "Drag and drop or ",
                                html.A("Select Files", href="#"),
                            ]
                        ),
                        style=upload_style,
                        # Allow multiple files to be uploaded
                        multiple=True,
                    ),
                    id="upload-replace-column",
                ),
                dbc.Col(
                    dcc.Upload(
                        id="upload-data-add",
                        children=html.Div(
                            [
                                html.Strong("Add scenarios"),
                                html.Br(),
                                "Drag and drop or ",
                                html.A("Select Files", href="#"),
                            ]
                        ),
                        style=upload_style,
                        multiple=True,
                    )
                ),
            ]
        )
    ],
    style={"padding-top": "10px"},
)


@app.callback(
    Output("upload-replace-column", "style"),
    Input("plot-data-store", "data"),
)
def show_replace_upload(databases):
    return {} if databases else {"display": "none"}


layout_dropdown = dcc.Dropdown(
    id="plot-fileselection-filename",
    options=all_experiments_options(),
    multi=True,
)
layout_refreshbutton = dbc.Button(
    "Refresh", color="secondary", id="plot-fileselection-refresh"
)

layout_select = html.P(
    [dbc.Row([dbc.Col([layout_dropdown]), dbc.Col([layout_refreshbutton], md=2)])],
    style={"padding-top": "10px"},
)

style = {"active_label_style": {"color": "black"}, "label_style": {"color": "#89a041"}}

layout = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    layout_upload, label="Upload files", tab_id="files-upload", **style
                ),
                dbc.Tab(
                    layout_select,
                    label="Select existing scenarios",
                    tab_id="files-select",
                    **style,
                ),
            ],
            id="file-select-type",
            active_tab="files-upload",
        ),
        html.Div(id="output-data-upload"),
        html.Div(id="output-data-upload2"),
    ]
)


def handle_upload_update(list_of_content, list_of_names):
    params = {}
    parsed_data = []
    if list_of_content is not None:
        i = 0
        for content, filename in zip(list_of_content, list_of_names):
            line_dash = data.LINE_STYLES[i % len(data.LINE_STYLES)]
            parsed_dict, is_data = parse_contents(content, filename, line_dash)
            if parsed_dict is not None:
                if is_data:
                    parsed_data.append((filename, parsed_dict))
                    i += 1
                else:
                    params[filename] = parsed_dict

    for filename, parsed_dict in parsed_data:
        params_filename = f"{filename}.params.json"
        if params_filename in params:
            parsed_dict["meta"]["params"] = params[params_filename]

    return data.label_scenarios(parsed_data)


def add_databases(databases, additions):
    """Add new scenarios and consistently relabel the combined collection."""
    combined = {}
    for name, database in (databases or {}).items():
        filename = database.get("meta", {}).get("filename")
        if filename is None:
            filename = name.rsplit(": ", 1)[-1]
        combined[filename] = database

    for name, database in (additions or {}).items():
        filename = database["meta"]["filename"]
        combined[filename] = database

    return data.label_scenarios(list(combined.items()))


def handle_select_update(names):
    # Put database in cache
    databases = data.dataStore.get(names, as_dict=True)
    if databases is None or len(databases) == 0:
        raise PreventUpdate
    return databases


@app.callback(
    Output("plot-data-store", "data"),
    [
        Input("file-select-type", "active_tab"),
        Input("plot-fileselection-filename", "value"),
        Input("upload-data", "contents"),
        Input("upload-data-add", "contents"),
        State("upload-data", "filename"),
        State("upload-data-add", "filename"),
        State("plot-data-store", "data"),
    ],
)
def update_data(
    active_tab,
    select_names,
    upload_contents,
    add_contents,
    upload_names,
    add_names,
    databases,
):
    if active_tab == "files-select":
        return handle_select_update(select_names)
    if active_tab == "files-upload":
        if dash.ctx.triggered_id == "upload-data-add":
            additions = handle_upload_update(add_contents, add_names)
            if additions is None:
                raise PreventUpdate
            return add_databases(databases, additions)
        if dash.ctx.triggered_id == "upload-data":
            replacements = handle_upload_update(upload_contents, upload_names)
            if replacements is None:
                raise PreventUpdate
            return replacements
        return databases
    raise PreventUpdate


# @app.callback(
#     Output("plot-data-store", "data"), [Input("plot-fileselection-filename", "value")],
# )
# def update_data_store(names):
#     # Put database in cache
#     databases = data.dataStore.get(names, as_dict=True)
#     if databases is None or len(databases) == 0:
#         raise PreventUpdate
#     print(databases.keys())
#     return databases


@app.callback(
    Output("plot-fileselection-filename", "options"),
    [Input("plot-fileselection-refresh", "n_clicks")],
)
def update_options(n):
    if n is None or n == 0:
        raise PreventUpdate
    data.dataStore.reset()
    return all_experiments_options()


@app.callback(
    Output("plot-fileselection-filename", "value"),
    [Input("plot-fileselection-refresh", "n_clicks")],
)
def clear_filename(n):
    if n is None or n == 0:
        raise PreventUpdate
    return []


##################
#
# Upload files
#
##################


def parse_contents(content, filename: str, line_dash):
    content_type, content_string = content.split(",")

    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith(".csv"):
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith(".json"):
            params = json.loads(decoded.decode("utf-8"))
            return params, False
        else:
            return None, None
    except Exception as e:
        print(e)
        return None, None  # TODO
        # return html.Div(["There was an error processing this file."])

    # In old versions of MIMOSA, baseline_emissions was called "baseline", and "ssp_baseline_emissions" was called "baseline_emissions"
    variables = df["Variable"].unique()
    if "baseline" in variables:
        df = df[df["Variable"] != "baseline_emissions"]
    return {"data": df.to_dict(), "meta": {"line_dash": line_dash}}, True
