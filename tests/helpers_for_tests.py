from builtins import range
import numbers
import subprocess
import os
import numpy as np
from pytest import approx
import urllib
import shutil
from pynwb import NWBHDF5IO, validate



def compare_dicts(d_ref, d):
    """
    Custom test assertion for dictionaries with a mixture of strings,
    floating point values and ndarrays of these.

    pytest does not support passing in dicts of numpy arrays with strings.

    See https://github.com/pytest-dev/pytest/issues/4079.

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
                if isinstance(array[index], (str, bytes)):
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


def diff_h5(test_file,temp_file):
    """
    Compare h5 files

    Parameters
    ----------
    test_file: str nwb file name
    temp_file: str nwb file name

    Returns
    -------
    int h5diff exit code:     0 if no differences, 1 if differences found, 2 if error
    """

    prog_args = ["-c", "-v2", "--follow-symlinks", "--no-dangling-links"]

    # list of objects to ignore
    # these objects always change
    ignore_paths = ["--exclude-path", "/general/source_script",
                    "--exclude-path", "/file_create_date",
                    "--exclude-path", "/identifier",
                    "--exclude-path", "/specifications"]

    nwb_files = [test_file, temp_file]

    # MSYS_NO_PATHCONV is for users who are running the tests in a MSYS
    # bash on windows.
    # See https://stackoverflow.com/a/34386471/4859183 for some background.

    out = subprocess.run(["h5diff"] + prog_args + ignore_paths + nwb_files,
                         env={"MSYS_NO_PATHCONV": "1"},
                         encoding="ascii",
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

    print(out.stdout)
    print(out.stderr)

    return out.returncode


def validate_nwb(filename):
    """
    If pynwb does not catch an exception then add it to the error list

    Parameters
    ----------
    filename: str nwb file name

    Returns
    -------
    errors: list of errors
    """

    if os.path.exists(filename):
        with NWBHDF5IO(filename, mode='r', load_namespaces=True) as io:
            try:
                errors = validate(io)
            except Exception as e:
                errors = [e]

    return errors
