"""
Common Dash objects
"""

# pylint: disable=unused-import
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


def short_name(filename):
    return filename.split(":")[0] if ":" in filename else filename
