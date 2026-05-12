"""
Provides the outer page framework, as well as the routing functionality.
"""

from common.dash import dcc, dbc, html, Input, Output

from app import app
from pages import plot

server = app.server


app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content"),
        dcc.Store(id="common-lastmodified-store"),
    ],
    className="p-5",
    fluid=True,
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
# pylint: disable=unused-argument
def display_page(pathname):
    return plot.layout
    # if pathname == '/settings':
    #     return settings.layout
    # else:
    #     return main.layout


if __name__ == "__main__":
    app.run(debug=True)
