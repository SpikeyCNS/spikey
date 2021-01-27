"""
Test writing and reading of generic log files in logger/.
"""
import os

import pytest
import pandas as pd

from spikey.logging import log, Reader


FILENAME = "test_log.json"
RESULTS = {
    "result1": 1,
    "result2": 2,
}
INFO = {
    "info1": None,
    "info2": "example.",
}


@pytest.fixture
def write_log():
    try:
        log(
            None,
            None,
            results=RESULTS,
            info=INFO,
            folder="",
            filename=FILENAME,
        )

    except Exception as e:
        yield e
    else:
        yield None

    os.remove(FILENAME)


def test_write_log(write_log):
    error = write_log
    assert error is None, f"Error when writing file! '{error}'"

    ## Filename exist?
    assert os.path.exists(FILENAME), "File doesnt exist!"

    ## File has data?
    assert os.stat(FILENAME).st_size > 40, "File has no contents."


def test_read_log(write_log):
    reader = Reader(folder=".", filenames=[FILENAME])

    ## Test df
    output_df = reader.df
    expected = pd.DataFrame([list(RESULTS.values())], columns=list(RESULTS.keys()))
    assert isinstance(output_df, pd.DataFrame)
    pd.testing.assert_frame_equal(output_df, expected, check_dtype=False)

    ## Test __getitem__
    for SET in [INFO, RESULTS]:
        for key, value_expected in SET.items():
            value_real = reader[key][0]

            assert (
                value_expected == value_real
            ), f"{key}: {value_expected} != {value_real}"
