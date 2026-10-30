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
        html.H4("Residual vs gross damage costs, by impact sector"),
        dcc.Loading(html.Div(id="tabs-damages-main-plots")),
        html.H4("Adaptation level, by impact sector"),
        dcc.Loading(
            dcc.Graph(id="tabs-damages-adaptlevel-plot", className="fixed_height_plot")
        ),
        html.H4("Sea level rise:"),
        dcc.Loading(html.Div(id="tabs-damages-SLR-plots")),
    ],
    className="tabs-plot-container",
)


## Residual vs gross damage costs
@app.callback(
    Output("tabs-damages-main-plots", "children"),
    [Input("plot-selected-store", "data"), Input("plot-timerange", "value")],
)
def update_damages_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        return []

    height = max(250, params.DEFAULT_PLOT_HEIGHT / len(databases))
    figs = [
        html.Div(
            [
                html.Strong(short_name(filename) + ":"),
                dcc.Graph(
                    figure=plotutils.create_plot(
                        {filename: single_df},
                        [
                            "non_slr_damage_costs",
                            "slr_damage_costs",
                            "slr_damage_costs_gross",
                            "labourprod_damage_costs",
                            "labourprod_damage_costs_net",
                            "labourprod_damage_costs_gross",
                            "riverine_damage_costs",
                            "riverine_damage_costs_gross",
                        ],
                        timerange,
                        stackgroup={
                            "non_slr_damage_costs": "costs",
                            "slr_damage_costs": "costs",
                            "riverine_damage_costs": "costs",
                            "labourprod_damage_costs_net": "costs",
                        },
                        yaxis_title="Costs",
                        tickformat="p",
                        height=height,
                        hidden_variables=[
                            "slr_damage_costs_gross",
                            "labourprod_damage_costs",
                            "labourprod_damage_costs_gross",
                            "riverine_damage_costs_gross",
                        ],
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
    [Input("plot-selected-store", "data"), Input("plot-timerange", "value")],
)
def update_adaptlevel_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        return {}

    fig = plotutils.create_plot(
        databases,
        [
            "labourprod_adaptation_level",
            "riverine_adaptation_level",
            "slr_adaptation_level",
        ],
        timerange,
        colors=[4, 5, 6],
        height=params.FIXED_PLOT_HEIGHT,
    )
    return fig


## Sea level rise
@app.callback(
    Output("tabs-damages-SLR-plots", "children"),
    [Input("plot-selected-store", "data"), Input("plot-timerange", "value")],
)
def update_slr_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        return []

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
                                    "slr_thermal",  # New names
                                    "slr_cumgsic",
                                    "slr_cumgis",
                                    "slr_cumais",
                                    "slr_cumlws",
                                    "total_SLR",
                                ],
                                timerange,
                                stackgroup={
                                    "slr_thermal": "SLR",
                                    "slr_cumgsic": "SLR",
                                    "slr_cumgis": "SLR",
                                    "slr_cumais": "SLR",
                                    "slr_cumlws": "SLR",
                                },
                                yaxis_title="SLR",
                                colors=[8, 9, 10, 4, 11, 6],
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
