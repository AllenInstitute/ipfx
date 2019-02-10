import numbers

import numpy as np
from pytest import approx
from six.moves import urllib
import six
import shutil


def compare_dicts(d_ref, d):
    """
    Custom test assertion for dictionaries with a mixture of strings,
    floating point values and ndarrays of these.

    pytest does not support passing in dicts of numpy arrays with strings.

    See https://github.com/pytest-dev/pytest/issues/4079 and
    https://github.com/pytest-dev/pytest/issues/4079.

    And dictdiffer does not work for our NaNs mixture see
    https://github.com/inveniosoftware/dictdiffer/issues/114.
    """

    assert sorted(d_ref.keys()) == sorted(d.keys())
    for k, v in d_ref.items():
        if isinstance(v, np.ndarray):
            array_ref = d_ref[k]
            array = d[k]

            assert len(array) == len(array_ref)
            for index in range(len(array)):
                if isinstance(array[index], (str, six.text_type)):
                    assert array[index] == array_ref[index]
                else:
                    assert array[index] == approx(array_ref[index])
        else:
            value_ref = d_ref[k]
            value = d[k]

            if isinstance(value_ref, numbers.Number):
                assert value_ref == approx(value, nan_ok=True)
            else:
                assert value_ref == value


def download_file(file_name, output_filepath):
    """
    Download the file pointed to by `url` and store it in
    `output_filepath`.
    """

    BASE_URL = "https://www.byte-physics.de/Downloads/allensdk-test-data/"

    response = urllib.request.urlopen(BASE_URL + file_name)
    with open(output_filepath, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
