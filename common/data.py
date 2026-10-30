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


def filename_stem(filename):
    """Remove a supported data-file extension from a filename."""
    stem, extension = os.path.splitext(filename)
    if extension.lower() in {".csv", ".xls", ".xlsx"}:
        return stem
    return filename


def scenario_shortnames(filenames):
    """Return filename stems without common underscore-delimited name parts.

    Only complete tokens are removed. The suffix is calculated after the prefix
    has been removed, so the two can never overlap. A single filename keeps its
    complete stem.
    """
    stems = [filename_stem(filename) for filename in filenames]
    if len(stems) < 2:
        return stems

    token_lists = [stem.split("_") for stem in stems]

    prefix_length = 0
    for tokens in zip(*token_lists):
        if len(set(tokens)) != 1:
            break
        prefix_length += 1

    without_prefix = [tokens[prefix_length:] for tokens in token_lists]

    suffix_length = 0
    for tokens in zip(*(tokens[::-1] for tokens in without_prefix)):
        if len(set(tokens)) != 1:
            break
        suffix_length += 1

    if suffix_length:
        shortened_tokens = [tokens[:-suffix_length] for tokens in without_prefix]
    else:
        shortened_tokens = without_prefix

    shortnames = ["_".join(tokens) for tokens in shortened_tokens]

    # This can only occur for duplicate names (or names consisting entirely of
    # shared text). Keeping the stem is more useful than displaying an empty name.
    return [shortname or stem for shortname, stem in zip(shortnames, stems)]


def filename_to_path(filename):
    # Strip filename of slashes for security
    filename = filename.replace("/", "").replace("\\", "")
    return os.path.join(os.path.dirname(__file__), "..", DATA_DIRECTORY, filename)


# Get all available experiments
def get_all_experiments():
    paths = glob.glob(filename_to_path("*.csv"))
    basenames = [os.path.os.path.basename(path) for path in paths]
    basenames.sort()
    return basenames


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

    def get(self, filenames, params: bool = False, as_dict: bool = False):
        if filenames is None:
            return None
        # For each filename in the list, get the corresponding database
        loaded = []
        for filename, line_dash in zip(filenames, LINE_STYLES):
            database = self.get_single_data(filename, as_dict)
            params = self.get_single_params(filename)
            if database is not None:
                loaded.append((filename, line_dash, database, params))

        databases = {}
        shortnames = scenario_shortnames([filename for filename, *_ in loaded])
        for i, (shortname, (filename, line_dash, database, params)) in enumerate(
            zip(shortnames, loaded)
        ):
            letter = chr(ord("A") + i)
            name = f"Scenario {letter} ({shortname}): {filename_stem(filename)}"
            databases[name] = {
                "data": database,
                "meta": {"line_dash": line_dash, "params": params},
            }
        if len(databases) == 0:
            return None
        return databases

    def get_single_data(self, filename, as_dict: bool):
        # 1. Check if file still needs to be read out
        if filename not in self.databases:
            self.databases[filename] = read_content(filename)
        # 2. Return database from cache
        if as_dict:
            return self.databases[filename].to_dict()
        return self.databases[filename]

    def get_single_params(self, filename) -> dict:
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
