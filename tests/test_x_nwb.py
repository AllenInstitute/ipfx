"""
Regression tests for the DAT/ABF conversion to NWB.

Idea:
    A list of raw data files is converted to NWB and the new NWB file
    is compared to the NWB from earlier runs.


Running the tests:
    By default these tests are expensive and thus skipped by default.
    To run them pass `--do-x-nwb-tests` on the command line. Tests execution
    can be parallelized via passing `--numprocesses auto`.
"""

import pytest
import glob
import os
import subprocess

import pyabf

from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.utils import createCycleID
from ipfx.bin.run_x_to_nwb_conversion import convert
from .test_x_nwb_helper import fetch_and_extract_zip


if not pytest.config.getoption("--do-x-nwb-tests"):
    pytest.skip("--do-x-nwb-tests is missing, skipping tests",
                allow_module_level=True)


def get_raw_files():

    # we have to do that here as we need to have the files
    # before we can decide how many tests we have
    fetch_and_extract_zip("reference_dat.zip")
    fetch_and_extract_zip("reference_dat_nwb.zip")

    fetch_and_extract_zip("reference_abf.zip")
    fetch_and_extract_zip("reference_abf_nwb.zip")

    fetch_and_extract_zip("reference_atf.zip")
    ABFConverter.protocolStorageDir = "reference_atf"

    files = []

    for ext in ["abf", "dat"]:
        folder = f"reference_{ext}"
        files += glob.glob(os.path.join(folder, f"*.{ext}"))

    return files


@pytest.fixture(scope="module")
def h5diff_present():
    assert subprocess.run(["h5diff", "--version"]).returncode == 0


@pytest.fixture(scope="module", params=get_raw_files())
def raw_file(request, h5diff_present):
    return request.param


def test_file_level_regressions(raw_file):

    base, ext = os.path.splitext(raw_file)

    ref_folder = f"reference_{ext[1:]}_nwb"

    new_file = convert(raw_file, overwrite=True, outputFeedbackChannel=True)
    ref_file = os.path.join(ref_folder, os.path.basename(new_file))

    assert os.path.isfile(ref_file)
    assert os.path.isfile(new_file)

    prog_args = ["-c", "-v2", "--follow-symlinks", "--no-dangling-links"]

    # list of objects to ignore
    # these objects always change
    ignore_paths = ["--exclude-path", "/general/source_script",
                    "--exclude-path", "/file_create_date",
                    "--exclude-path", "/identifier",
                    "--exclude-path", "/specifications"]

    nwb_files = [ref_file, new_file]

    # MSYS_NO_PATHCONV is for users how are running the tests in a MSYS
    # bash on windows.
    # See https://stackoverflow.com/a/34386471/4859183 for some background.
    out = subprocess.run(["h5diff"] + prog_args + ignore_paths + nwb_files,
                         env={"MSYS_NO_PATHCONV": "1"}, encoding="ascii",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # h5diff exit codes:
    # 0 if no differences, 1 if differences found, 2 if error
    if out.returncode:
        print(out.stdout)

    assert out.returncode == 0


def test_createCycleID():

    assert createCycleID([1, 2, 3, 4], total=2)  == 1234
    assert createCycleID([1, 2, 3, 4], total=20) == 1020304
    assert createCycleID([10, 2, 3, 4], total=20) == 10020304
    assert createCycleID([10, 2, 3, 40], total=20) == 10020340
    assert createCycleID([10, 20, 30, 4], total=20) == 10203004
    assert createCycleID([10, 20, 30, 40], total=20) == 10203040
