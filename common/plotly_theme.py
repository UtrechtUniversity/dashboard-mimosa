import plotly.io as pio
import plotly.graph_objects as go


PLOTLY_WIDTH_PX = 1050
PAGE_WIDTH_MM = 170.65


def mm_to_px(mm, width_mm=PAGE_WIDTH_MM, width_px=PLOTLY_WIDTH_PX):
    return mm / width_mm * width_px


def pt_to_px(pt):
    mm = 0.353 * pt
    return mm_to_px(mm)


SVG_DPI = 96  # This is default from Plotly export, we cannot change that
SVG_SCALE = (PAGE_WIDTH_MM / 25.4) / (
    PLOTLY_WIDTH_PX / SVG_DPI
)  # page width in inch (div. by 25.4)

FRUTIGER = '"Open Sans", "Frutiger LT Pro Condensed",verdana, arial, sans-serif'

IPCC_COLORS = [
    "#5492cd",
    "#ffa900",
    "#003466",
    "#EF550F",
    "#990002",
    "#c47900",
    "#00aad0",
    "#76797b",
]

CATEGORIES_COLORS = [
    "#97CEE4",
    "#778663",
    "#6F7899",
    "#A7C682",
    "#8CA7D0",
    "#FAC182",
    "#F18872",
    "#bd7161",
]

pio.templates["ipcc"] = go.layout.Template(
    {
        "data": dict(
            scatter=[
                go.Scatter(marker_color=color, line_color=color)
                for color in IPCC_COLORS
            ],
            heatmap=[go.Heatmap(colorbar_outlinewidth=0)],
        ),
        "layout": dict(
            title={
                "font_family": FRUTIGER,
                "font_size": pt_to_px(11),
                "xanchor": "left",
                "x": 0.01,
            },
            plot_bgcolor="#F7F7F7",
            font={
                "family": FRUTIGER,
                "size": pt_to_px(7),
                "color": "#25292b",
            },
            legend={
                "font_size": pt_to_px(7),
                "title_font_size": pt_to_px(7),
                "y": 0.5,
                "tracegroupgap": 0,
            },
            colorway=IPCC_COLORS,
            coloraxis={"colorbar": {"outlinewidth": 0, "ticks": ""}},
            barmode="relative",
            xaxis=dict(
                title_font_size=pt_to_px(8),
                title_standoff=0,
                gridcolor="#d0d0d0",
                gridwidth=pt_to_px(0.25),
                zerolinecolor="#8e8e8d",
                zerolinewidth=2,
                ticks="",
                automargin=True,
                showgrid=False,
            ),
            yaxis=dict(
                title_font_size=pt_to_px(8),
                title_standoff=0,
                ticksuffix=" ",
                gridcolor="#d0d0d0",
                gridwidth=pt_to_px(0.25),
                zerolinecolor="#8e8e8d",
                zerolinewidth=2,
                ticks="",
                automargin=True,
            ),
            margin={"t": 60, "r": 50},
        ),
    }
)
