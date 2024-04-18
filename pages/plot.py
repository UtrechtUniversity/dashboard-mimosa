"""
The Plot-page consists of the file selection bar, and the tabs containing all the plots
"""

from common.dash import html, dbc

from pages.components import fileselection, legend, tabs

# back_button = dbc.Button("< Back to content", href="/", color="primary", outline=True)

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.A(
                            html.Img(
                                src="assets/MIMOSA.svg",
                                width=350,
                                style={"marginTop": "5px"},
                            ),
                            href="https://utrechtuniversity.github.io/mimosa",
                            target="_blank",
                        )
                    ],
                    style={"flex": "400px 0 0"},
                ),
                dbc.Col(
                    [
                        html.H1("Dashboard"),
                        html.H3("Plot results"),
                    ]
                ),
            ]
        ),
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
