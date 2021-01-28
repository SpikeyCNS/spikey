"""
Logging functionaltiy.
"""
from datetime import datetime
import json
import os

import numpy as np

from spikey.logging.sanitize import sanitize_dictionary


def log(
    network: object,
    game: object,
    results: dict = None,
    info: dict = None,
    folder: str = "",
    filename: str = None,
):
    """
    Save network and game data.

    {
        'metadata': value,
        'snn': {
            Network configuration data.
        },
        'game': {
            Game configuration data.
        },
        'results': {
            Results, can be directly loaded to table.
        },
        'info': {
            Information to gather further information on.
            Not loaded in table by default.
        }
    }

    Parameters
    ----------
    network: SNN
        Network of interest.
    game: Game
        Played game.
    results: dict, default=None
        Custom results to log.
    info: dict, default=None
        Extra information to log.
    folder: str, default='snn/log'
        Change folder to save in.
    filename: str, default=None
        Filename to write(replaces whole path).

    Post
    ----
    Saves .json in folder

    Returns
    -------
    str Filename
    """
    if not folder and not filename:
        folder = os.path.join(os.path.abspath(__file__).split("spikey")[0], "log")

    if folder:
        try:
            os.makedirs(folder)

            print(f"Created directory {folder}!")
        except FileExistsError:
            pass

    if filename is None:
        filename = os.path.join(
            folder, f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}.json"
        )
    else:
        filename = os.path.join(folder, filename)

    data = {}

    if game is not None:
        ## Game Parameters
        game_params = {
            "name": str(type(game)),
        }
        game_params.update(game.params)
        data.update({"game": sanitize_dictionary(game_params)})

    if network is not None:
        ## Network Parameters
        snn_params = {}
        for key, value in network.parts.items():
            if hasattr(value, "__name__"):
                snn_params.update({key: value.__name__})
            else:
                snn_params.update({key: str(type(value).__name__)})

        snn_params.update(network.params)

        data.update({"snn": sanitize_dictionary(snn_params)})

    data = sanitize_dictionary(data)

    if results is not None:
        data.update({"results": sanitize_dictionary(results)})
    if info is not None:
        data.update({"info": sanitize_dictionary(info)})

    order_rankings = {
        int: 0,
        float: 0,
        str: 20,
        tuple: 40,
        "default": 50,
        list: 60,
        set: 70,
        np.ndarray: 99,
        np.ma.core.MaskedArray: 99,
    }

    for data_key, values in data.items():
        evaluations = {}

        if not isinstance(values, dict):
            continue

        for key, value in values.items():
            value_type = type(value)

            if value_type not in order_rankings:
                evaluations[key] = 50
            else:
                evaluations[key] = order_rankings[value_type]

        ordering = sorted(evaluations, key=evaluations.get)
        data[data_key] = {key: values[key] for key in ordering}

    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

    return filename
