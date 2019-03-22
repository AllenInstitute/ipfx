"""
Regenerating NWB files:

    cd tests
    rm *zip *sha256
    python test_x_nwb_recreate_files.py
    # upload the zip/sha256 files

Notes:
    The reference DAT package file has only data acquired with
    Patchmaster 2x90.

    The list of files was generated with:
    grep "\bVersion" *pymeta | grep 2x90 | uniq | cut -f 1 -d ":" \
    | sed -s "s/.pymeta//g" | xargs zip -u reference_dat_files.zip

    pymeta files are created by `hr_bundle.py`.
"""

import shutil
import hashlib
import os
import glob
from zipfile import ZipFile, ZIP_DEFLATED

from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.bin.run_x_to_nwb_conversion import convert
from tests.helpers_for_tests import download_file


def fetch_and_extract_zip(filename):
    """
    Download regression test data.

    Files which must reside in `BASE_URL`:
    BASE_URL + filename: ZIP file, will be extracted into `filename`
                         without extension on success
    BASE_URL + filename + ".sha256": Holds the SHA256 hash of the ZIP file

    The usage of the hash file allows us to skip downloading the ZIP file if it
    is already present and in the current version.
    """

    def compare_checksums(path, checksum):
        """
        Return `True` if `path` has the SHA256 checksum given in `checksum`.

        checksum_file is expected to have a format compatible to the
        `sha256sum` commandline tool.

        Format:
        ```
            3bd33[...] *reference_dat_files.zip
        ```
        """

        with open(path, "rb") as f:
            existing = hashlib.sha256(f.read()).hexdigest()

        with open(checksum, "rb") as f:
            expected = f.read().decode("ascii").strip().split(" ")[0]

        return existing == expected

    checksum = filename + ".sha256"
    download_file(checksum, checksum)

    folder = os.path.splitext(filename)[0]
    needs_extract = not os.path.isdir(folder)

    if (os.path.isfile(filename)
        and not compare_checksums(filename, checksum)) \
       or not os.path.isfile(filename):  # file not present
        print(f"Download large file {filename}, please be patient.")
        download_file(filename, filename)
        needs_extract = True

        if not compare_checksums(filename, checksum):
            raise ValueError("File and its checksum don't match!")

    if needs_extract:
        with ZipFile(filename, "r") as f:
            print(f"Extracting {filename} into {folder}, please be patient.")
            f.testzip()
            f.extractall(folder)


def create_files_for_upload(ext):
    """
    Create the zip files and their checksum files for the given format.
    These files can be uploaded directly to `BASE_URL`.
    """

    fetch_and_extract_zip(f"reference_{ext}.zip")

    if ext == "abf":
        fetch_and_extract_zip("reference_atf.zip")
        ABFConverter.protocolStorageDir = "reference_atf"

    folder = f"reference_{ext}"
    basename = f"reference_{ext}"
    zip_files(folder, basename, f".{ext}")

    if ext == "atf":
        return None

    files = glob.glob(os.path.join(folder, "*." + ext))

    for f in files:
        print(f"Converting {f}")
        convert(f, overwrite=True, outputFeedbackChannel=True, multipleGroupsPerFile=True)

    nwb_folder = basename + "_nwb"
    zip_files(folder, nwb_folder, ".nwb")
    shutil.rmtree(nwb_folder, ignore_errors=True)


def zip_files(folder, filename, extension):
    """
    Zip all files matching `extension` in `folder` and save them into
    `filename + ".zip"` and create a checksum file as well.
    """

    if not os.path.isdir(folder):
        raise ValueError(f"{folder} needs to be an existing folder.")

    cwd = os.getcwd()

    try:
        os.chdir(folder)
        files = glob.glob("*" + extension)

        if len(files) == 0:
            raise ValueError(f"Could not find any files in {folder}")

        filename = os.path.join("..", filename) + ".zip"
        checksum = filename + ".sha256"

        with ZipFile(filename, "w", compression=ZIP_DEFLATED) as z:
            for f in files:
                z.write(f)

        with open(filename, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            with open(checksum, "wb") as f:
                content = f"{h} *{os.path.basename(filename)}\n"
                f.write(content.encode("ascii"))
    finally:
        os.chdir(cwd)


def main():
    create_files_for_upload("dat")
    # create_files_for_upload("atf")
    create_files_for_upload("abf")
    pass


if __name__ == "__main__":
    main()
