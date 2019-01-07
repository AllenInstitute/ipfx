#!/bin/env python

import os
import argparse

from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.DatConverter import DatConverter


def convert(inFileOrFolder, overwrite=False, fileType=None, outputMetadata=False, outputFeedbackChannel=False):
    """
    Convert the given file to a NeuroDataWithoutBorders file using pynwb

    Supported fileformats:
        - ABF v2 files created by Clampex
        - DAT files created by Patchmaster v2x90

    :param inFileOrFolder: path to a file or folder
    :param overwrite: overwrite output file, defaults to `False`
    :param fileType: file type to be converted, must be passed iff `inFileOrFolder` refers to a folder
    :param outputMetadata: output metadata of the file, helpful for debugging
    :param outputFeedbackChannel: Output ADC data which stems from stimulus feedback channels (ignored for DAT files)

    :return: path of the created NWB file
    """

    if not os.path.exists(inFileOrFolder):
        raise ValueError(f"The file {inFileOrFolder} does not exist.")

    if os.path.isfile(inFileOrFolder):
        root, ext = os.path.splitext(inFileOrFolder)
    if os.path.isdir(inFileOrFolder):
        if not fileType:
            raise ValueError("Missing fileType when passing a folder")

        inFileOrFolder = os.path.normpath(inFileOrFolder)
        inFileOrFolder = os.path.realpath(inFileOrFolder)

        ext = fileType
        root = os.path.join(inFileOrFolder, "..",
                            os.path.basename(inFileOrFolder))

    outFile = root + ".nwb"

    if not outputMetadata and os.path.exists(outFile):
        if overwrite:
            os.remove(outFile)
        else:
            raise ValueError(f"The output file {outFile} does already exist.")

    if ext == ".abf":
        if outputMetadata:
            ABFConverter.outputMetadata(inFileOrFolder)
        else:
            ABFConverter(inFileOrFolder, outFile, outputFeedbackChannel=outputFeedbackChannel)
    elif ext == ".dat":
        if outputMetadata:
            DatConverter.outputMetadata(inFileOrFolder)
        else:
            DatConverter(inFileOrFolder, outFile)

    else:
        raise ValueError(f"The extension {ext} is currently not supported.")

    return outFile


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite the output NWB file")
    parser.add_argument("--protocolDir", type=str,
                        help=("Disc location where custom waveforms "
                              "in ATF format are stored."))
    parser.add_argument("--fileType", type=str, default=None, choices=[".abf"],
                        help=("Type of the files to convert (only required "
                              "if passing folders)."))
    parser.add_argument("--outputMetadata", action="store_true", default=False,
                        help="Helper for debugging which outputs HTML/TXT files with the metadata contents of the files.")
    parser.add_argument("--outputFeedbackChannel", action="store_true", default=False,
                        help="Output ADC data to the NWB file which stems from stimulus feedback channels.")
    parser.add_argument("filesOrFolders", nargs="+",
                        help="List of ABF files/folders to convert.")

    args = parser.parse_args()

    if args.protocolDir:
        if not os.path.exists(args.protocolDir):
            raise ValueError("Protocol directory does not exist")

        ABFConverter.protocolStorageDir = args.protocolDir

    for fileOrFolder in args.filesOrFolders:
        print(f"Converting {fileOrFolder}")
        convert(fileOrFolder, overwrite=args.overwrite, fileType=args.fileType,
                outputMetadata=args.outputMetadata,
                outputFeedbackChannel=args.outputFeedbackChannel)


if __name__ == "__main__":
    main()
