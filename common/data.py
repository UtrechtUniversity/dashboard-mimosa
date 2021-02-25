"""
Common data functions:
 - get_all_experiments
 - filename_to_path (only used internally)
and the class DataStore.
"""

import glob
import json
import os.path
import pandas as pd


DATA_DIRECTORY = "outputdata"
LINE_STYLES = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]


def filename_to_path(filename):
    # Strip filename of slashes for security
    filename = filename.replace("/", "").replace("\\", "")
    return os.path.join(os.path.dirname(__file__), "..", DATA_DIRECTORY, filename)


# Get all available experiments
def get_all_experiments():
    paths = glob.glob(filename_to_path("output_*.csv"))
    return [os.path.os.path.basename(path) for path in paths]


class DataStore:
    """Provides the databases for each filename.

    Each time a file is requested by the front end,
    this class checks if it was already read from the filesystem.
    If not, it saves the data to memory in the dict `databases`.
    The subsequent read is the instantaneous.

    Same for params files.
    """

    databases = {}
    params = {}

    def __init__(self):
        pass

    def reset(self):
        self.databases = {}
        self.params = {}

    def get(self, filenames, params: bool = False):
        if filenames is None:
            return None
        # For each filename in the list, get the corresponding database
        databases = {}
        for filename, line_dash in zip(filenames, LINE_STYLES):
            database = (
                self.get_single_params(filename)
                if params
                else self.get_single_data(filename)
            )
            if database is not None:
                databases[filename] = {
                    "data": database,
                    "meta": {"line_dash": line_dash},
                }
        if len(databases) == 0:
            return None
        return databases

    def get_single_data(self, filename):
        # 1. Check if file still needs to be read out
        if filename not in self.databases:
            self.databases[filename] = read_content(filename)
        # 2. Return database from cache
        return self.databases[filename]

    def get_single_params(self, filename):
        # 1. Check if param file exists and still needs to be read out
        if filename not in self.params:
            self.params[filename] = read_param(filename)
        # 2. Return param dict from cache
        return self.params[filename]


dataStore = DataStore()


# Read content of file
def read_content(filename):

    try:
        # First strip all slashes
        filename = filename.replace("/", "").replace("\\", "")

        content = pd.read_csv(filename_to_path(filename))
    except FileNotFoundError:
        return None

    return content


# Read content of parameter
def read_param(filename):

    try:
        # First strip all slashes
        filename = filename.replace("/", "").replace("\\", "") + ".params.json"

        with open(filename_to_path(filename), "r") as fh:
            content = json.load(fh)

    except FileNotFoundError:
        return None

    return content
