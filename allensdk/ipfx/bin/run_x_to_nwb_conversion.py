#!/bin/env python

import os
import sys
import argparse

import pyabf

from allensdk.ipfx.x_to_nwb.ABFConverter import ABFConverter


def convert(inFileOrFolder, overwrite=False, fileType=None):
    """
    Convert the given file to a NeuroDataWithoutBorders file using pynwb

    Supported fileformats:
        - ABF v2 files created by Clampex
    """

    if not os.path.exists(inFileOrFolder):
        raise ValueError(f"The file {inFileOrFolder} does not exist.")

    if os.path.isfile(inFileOrFolder):
        root, ext = os.path.splitext(inFileOrFolder)
    if os.path.isdir(inFileOrFolder):
        if not fileType:
            raise ValueError("Missing fileType when passing a folder")

        inFileOrFolder = os.path.normpath(inFileOrFolder)
        ext = fileType
        root = os.path.join(inFileOrFolder, "..", os.path.basename(inFileOrFolder))

    outFile = root + ".nwb"

    if os.path.exists(outFile):
        if overwrite:
            os.remove(outFile)
        else:
            raise ValueError(f"The output file {outFile} does already exist.")

    if ext == ".abf":
        ABFConverter(inFileOrFolder, outFile)
    else:
        raise ValueError(f"The extension {ext} is currently not supported.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite the output NWB file")
    parser.add_argument("--protocolDir", type=str,
                        help="Disc location where custom waveforms in ATF format are stored.")
    parser.add_argument("--fileType", type=str, default=None, choices=[".abf"],
                        help="Type of the files to convert (only required if passing folders).")
    parser.add_argument("filesOrFolders", nargs="+", help="List of ABF files/folders to convert.")

    args = parser.parse_args()

    if args.protocolDir:
        if not os.path.exists(args.protocolDir):
            raise ValueError("Protocol directory does not exist")

        ABFConverter.atfStorage = pyabf.ATFStorage(args.protocolDir)

    for fileOrFolder in args.filesOrFolders:
        print(f"Converting {fileOrFolder}")
        convert(fileOrFolder, overwrite=args.overwrite, fileType=args.fileType)


if __name__ == "__main__":
    main()
