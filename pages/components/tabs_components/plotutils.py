"""
Provides the function `create_plot` used in all the tabs.
This function loops over the provided databases (for different line dash styles),
then loops over the available regions as subplots and finally loops over the provided
variables in different colours.
"""

import time
import numpy as np
import pandas as pd
import plotly.io as pio

from common import params, plotly_theme


def create_plot(
    df_dict,
    variables,
    timerange,
    stackgroup=None,
    yaxis_title="",
    tickformat=None,
    height=params.DEFAULT_PLOT_HEIGHT,
    hidden_variables=None,
    colors=None,
    percapita=False,
):
    if stackgroup is None:
        stackgroup = {}

    traces = []

    for df_i, (name, df_info) in enumerate(df_dict.items()):
        database = df_info["data"]
        if isinstance(database, dict):
            # t0 = time.time()
            database = pd.DataFrame(database)
            # t1 = time.time()
            # print("Took {} seconds".format(t1 - t0))
        line_dash = df_info["meta"]["line_dash"]
        short_name = name.split(":")[0] if ":" in name else name

        selection = database[database["Variable"].isin(variables)]
        regions = list(selection["Region"].unique())

        regions_to_axis = {region: f"x{i+1}" for i, region in enumerate(regions)}

        if percapita:
            pop_selection = database[
                (database["Variable"] == "population")
                & (database["Region"].isin(regions))
            ]
            population_factor = pop_selection.drop(
                columns=["Variable", "Unit"], errors="ignore"
            ).set_index("Region")
            per_cap_unit = f"/({pop_selection.iloc[0]['Unit']})"
        else:
            population_factor = 1
            per_cap_unit = ""

        units = (
            selection[["Variable", "Unit"]]
            .drop_duplicates()
            .set_index("Variable")["Unit"]
            .replace(
                {
                    "usd*trillion/yr": "Trillion USD/yr",
                    "fraction_of_GDP": "% GDP",
                    "degC_above_PI": "°C above PI",
                }
            )
        )
        multiple_units = False
        if len(units.unique()) == 1:
            yaxis_title += f" ({units.iloc[0]}{per_cap_unit})"
        elif len(units) > 1:
            multiple_units = True

        var_i = 0
        for var_i, (variable, subselection) in enumerate(selection.groupby("Variable")):
            # Color list * 3 to get the same colours repeated three times,
            # such that it never runs out of range
            color = (plotly_theme.IPCC_COLORS * 3)[
                var_i if colors is None else colors[var_i]
            ]

            subselection = (
                subselection.drop(
                    columns=["Variable", "Unit"], errors="ignore"
                ).set_index("Region")
                / population_factor
            )

            unit_str = (
                f" ({units[variable]}{per_cap_unit})"
                if multiple_units and variable in units
                else ""
            )

            for region_i, (region, values) in enumerate(
                subselection.loc[:, str(timerange[0]) : str(timerange[1])].iterrows()
            ):
                traces.append(
                    {
                        "type": "scatter",
                        "x": [float(x) for x in values.index],
                        "y": list(values),
                        "xaxis": regions_to_axis[region],
                        "yaxis": "y1",
                        "line": {"color": color, "dash": line_dash},
                        "mode": "lines",
                        "name": variable + unit_str,
                        "legendgroup": variable,
                        "showlegend": region_i == 0 and df_i == 0,
                        "hovertemplate": f"{short_name}, {variable}: %{{y}}<extra></extra>",
                        "visible": (
                            "legendonly"
                            if hidden_variables is not None
                            and variable in hidden_variables
                            else None
                        ),
                        "stackgroup": stackgroup.get(variable),
                    }
                )
                # TODO: handle units automatically if available

    minyear = float(timerange[0])
    maxyear = float(timerange[-1])

    layout = {
        "grid": {"columns": len(regions), "rows": 1},
        "yaxis1": {"title": yaxis_title},
        "margin": {"l": 50, "r": 20, "t": 30, "b": 30},
        "legend": {
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.0,
            "yref": "container",
            "title": {"text": f"Variable{'s' if var_i > 0 else ''}:"},
        },
        "hovermode": "x",
        "height": height,
        "annotations": [
            {
                "text": region,
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "center",
                "yanchor": "middle",
                "x": (i + 0.5) / len(regions),
                "y": 1 + 0.05 * 450 / height,
                "font": {"size": 16},
            }
            for i, region in enumerate(regions)
        ],
        "shapes": [
            {
                "type": "line",
                "yref": "paper",
                "xref": f"x{i+1}",
                "x0": minyear,
                "x1": minyear,
                "y0": 0,
                "y1": 1,
                "line": {"width": 1, "color": "#666"},
            }
            for i in range(len(regions))
        ],
        "template": pio.templates["ipcc"],
    }

    if len(regions) > 5:
        for i in range(len(regions)):
            layout[f"xaxis{i+1}"] = {"nticks": 4}

    fig = {
        "data": traces,
        "layout": layout,
    }

    if tickformat is not None:
        for region_i in range(len(regions)):
            yaxis = f"yaxis{region_i+1}"
            if yaxis not in fig["layout"]:
                fig["layout"][yaxis] = {}
            fig["layout"][yaxis]["tickformat"] = tickformat

    return fig
