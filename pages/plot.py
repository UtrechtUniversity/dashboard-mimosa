"""
The Plot-page consists of the file selection bar, and the tabs containing all the plots
"""
from common.dash import html

from pages.components import fileselection, legend, tabs

# back_button = dbc.Button("< Back to content", href="/", color="primary", outline=True)

layout = html.Div(
    [
        html.H1("MIMOSA"),
        html.H3("Integrated Assessment Model – Plot results"),
        html.Hr(),
        fileselection.layout,
        legend.layout,
        html.Br(),
        tabs.layout,
        html.Br(),
        html.Div(id="empty", style={"display": "none"}),
        # html.Hr(),
        # back_button,
    ]
)
