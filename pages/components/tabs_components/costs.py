"""
Tab for the Cost plots:
  - stacked plots with costs
  - carbon prices
  - learning factors
"""

from common.dash import dcc, html, Input, Output, PreventUpdate, short_name
from common import data, params

from pages.components.tabs_components import plotutils

from app import app

layout = html.Div(
    [
        html.Br(),
        html.H4(
            "Regional relative adaptation costs, abatement costs and residual damages"
        ),
        html.Div(id="tabs-costs-plots"),
        html.H4("Regional carbon prices"),
        dcc.Loading(
            dcc.Graph(id="tabs-costs-carbonprices-plot", className="fixed_height_plot")
        ),
        html.H4("Learning factors"),
        dcc.Loading(
            dcc.Graph(id="tabs-costs-learning-plot", className="fixed_height_plot")
        ),
    ],
    className="tabs-plot-container",
)


## Regional costs (adaptation+abatement+resid damages)
@app.callback(
    Output("tabs-costs-plots", "children"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_costs_plot(databases, timerange):
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
                            "rel_abatement_costs",
                            "rel_mitigation_costs",
                            "damage_costs",
                            "rel_financial_transfer",
                        ],  # , "adapt_costs"],
                        timerange,
                        stackgroup={
                            "rel_abatement_costs": "costs",
                            "rel_mitigation_costs": "costs",
                            "damage_costs": "costs",
                            "rel_financial_transfer": "costs",
                            # "adapt_costs": "costs",
                        },
                        yaxis_title="Costs (% GDP)",
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


## Carbon prices
@app.callback(
    Output("tabs-costs-carbonprices-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_carbonprices_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["carbonprice"],
        timerange,
        yaxis_title="TODO",
        colors=[3],
        height=params.FIXED_PLOT_HEIGHT,
    )
    return fig


## Learning factors
@app.callback(
    Output("tabs-costs-learning-plot", "figure"),
    [Input("plot-data-store", "data"), Input("plot-timerange", "value")],
)
def update_learning_plot(databases, timerange):
    if databases is None or len(databases) == 0:
        raise PreventUpdate

    fig = plotutils.create_plot(
        databases,
        ["learning_factor", "LBD_factor", "LOT_factor"],
        timerange,
        colors=[4, 5, 6],
        height=params.FIXED_PLOT_HEIGHT,
    )
    return fig
