"""
Tab for the Damages plots:
 - Residual vs gross damages + adapt costs
 - Adaptation level
 - Stock vs flow adaptation
 - Sea level rise
"""

from common.dash import dcc, dbc, html, Input, Output, PreventUpdate, short_name
from common import data, params

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        html.H4("Residual vs gross damage costs, with adaptation costs"),
        dcc.Loading(html.Div(id="tabs-damages-main-plots")),
        # html.H4("Adaptation level (top) and stock vs flow (bottom):"),
        # dcc.Loading(
        #     dcc.Graph(id="tabs-damages-adaptlevel-plot", className="fixed_height_plot")
        # ),
        # dcc.Loading(
        #     dcc.Graph(id="tabs-damages-adaptcosts-plot", className="fixed_height_plot")
        # ),
        html.H4("Sea level rise:"),
        dcc.Loading(html.Div(id="tabs-damages-SLR-plots")),
    ],
    className="tabs-plot-container",
)


## Residual vs gross damage costs
@app.callback(
    Output("tabs-damages-main-plots", "children"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_damages_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    height = max(250, params.DEFAULT_PLOT_HEIGHT / len(databases))
    figs = [
        html.Div(
            [
                html.Strong(short_name(filename) + ":"),
                dcc.Graph(
                    figure=plotutils.create_plot(
                        {filename: single_df},
                        [
                            "adapt_costs",
                            "resid_damages",
                            "damage_costs_slr",
                            "damage_costs_non_slr",
                            "gross_damages",
                            "SLR_damages",
                        ],
                        timerange,
                        stackgroup={
                            "adapt_costs": "costs",
                            "resid_damages": "costs",
                            "damage_costs_slr": "costs",
                            "damage_costs_non_slr": "costs",
                        },
                        yaxis_title="Costs",
                        tickformat="p",
                        height=height,
                    ),
                    style={"height": f"{height}px"},
                ),
            ]
        )
        for filename, single_df in databases.items()
    ]
    return figs


## Adaptation level
@app.callback(
    Output("tabs-damages-adaptlevel-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_adaptlevel_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["adapt_level", "SLR_adapt_level"],
        timerange,
        colors=[4, 5],
        height=params.FIXED_PLOT_HEIGHT,
    )
    return fig


## Adaptation costs disaggregation
# @app.callback(
#     Output("tabs-damages-adaptcosts-plot", "figure"),
#     [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
# )
# def update_adaptcosts_plot(databases, timerange):
#     if databases is None or len(databases) == 0:
#         raise PreventUpdate

#     fig = plotutils.create_plot(
#         databases,
#         ["adapt_FAD", "adapt_IAD", "adapt_SAD"],
#         timerange,
#         hidden_variables=["adapt_SAD"],
#         height=params.FIXED_PLOT_HEIGHT,
#         yaxis_title="Costs (% GDP)",
#         tickformat="p",
#         colors=[5, 6, 7],
#     )
#     return fig


## Sea level rise
@app.callback(
    Output("tabs-damages-SLR-plots", "children"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_slr_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    height = params.DEFAULT_PLOT_HEIGHT
    figs = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.Strong(short_name(filename) + ":"),
                        dcc.Graph(
                            figure=plotutils.create_plot(
                                {filename: single_df},
                                [
                                    "SLR",  # Old names
                                    "CUMGSIC",
                                    "CUMGIS",
                                    "slr_thermal",  # New names
                                    "slr_cumgsic",
                                    "slr_cumgis",
                                    "total_SLR",
                                ],
                                timerange,
                                stackgroup={
                                    "SLR": "SLR",
                                    "CUMGSIC": "SLR",
                                    "CUMGIS": "SLR",
                                    "slr_thermal": "SLR",
                                    "slr_cumgsic": "SLR",
                                    "slr_cumgis": "SLR",
                                },
                                yaxis_title="SLR",
                                colors=[8, 9, 10, 11],
                                height=height,
                            ),
                            style={"height": f"{height}px"},
                        ),
                    ]
                )
            )
            for filename, single_df in databases.items()
        ]
    )
    return figs
