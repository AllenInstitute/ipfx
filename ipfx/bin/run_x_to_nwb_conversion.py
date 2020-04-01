#!/bin/env python

import os
import argparse
import logging

from ipfx.x_to_nwb.ABFConverter import ABFConverter
from ipfx.x_to_nwb.DatConverter import DatConverter


log = logging.getLogger(__name__)


def convert(inFileOrFolder, overwrite=False, fileType=None, outputMetadata=False, outputFeedbackChannel=False, multipleGroupsPerFile=False, compression=True):
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
    :param multipleGroupsPerFile: Write all Groups in the DAT file into one NWB
                                  file. By default we create one NWB per Group (ignored for ABF files).
    :param compression: Toggle compression for HDF5 datasets

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
            ABFConverter(inFileOrFolder, outFile, outputFeedbackChannel=outputFeedbackChannel, compression=compression)
    elif ext == ".dat":
        if outputMetadata:
            DatConverter.outputMetadata(inFileOrFolder)
        else:
            DatConverter(inFileOrFolder, outFile, multipleGroupsPerFile=multipleGroupsPerFile, compression=compression)

    else:
        raise ValueError(f"The extension {ext} is currently not supported.")

    return outFile


def main():

    parser = argparse.ArgumentParser()

    common_group = parser.add_argument_group(title="Common", description="Options which are applicable to both ABF and DAT files")
    abf_group = parser.add_argument_group(title="ABF", description="Options which are applicable to ABF")
    dat_group = parser.add_argument_group(title="DAT", description="Options which are applicable to DAT")

    feature_parser = common_group.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--compression', dest='compression', action='store_true', help="Enable compression for HDF5 datasets (default).")
    feature_parser.add_argument('--no-compression', dest='compression', action='store_false', help="Disable compression for HDF5 datasets.")
    parser.set_defaults(compression=True)

    common_group.add_argument("--overwrite", action="store_true", default=False,
                               help="Overwrite the output NWB file")
    common_group.add_argument("--outputMetadata", action="store_true", default=False,
                               help="Helper for debugging which outputs HTML/TXT files with the metadata contents of the files.")
    common_group.add_argument("--log", type=str, help="Log level for debugging, defaults to the root logger's value.")
    common_group.add_argument("filesOrFolders", nargs="+",
                               help="List of ABF files/folders to convert.")

    abf_group.add_argument("--protocolDir", type=str,
                            help=("Disc location where custom waveforms in ATF format are stored."))
    abf_group.add_argument("--fileType", type=str, default=None, choices=[".abf"],
                            help=("Type of the files to convert (only required if passing folders)."))
    abf_group.add_argument("--outputFeedbackChannel", action="store_true", default=False,
                        help="Output ADC data to the NWB file which stems from stimulus feedback channels.")
    abf_group.add_argument("--realDataChannel", type=str, action="append",
                        help=f"Define additional channels which hold non-feedback channel data. The default is {ABFConverter.adcNamesWithRealData}.")

    dat_group.add_argument("--multipleGroupsPerFile", action="store_true", default=False,
                           help="Write all Groups from a DAT file into a single NWB file. By default we create one NWB file per Group.")

    args = parser.parse_args()

    if args.log:
        numeric_level = getattr(logging, args.log.upper(), None)

        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {args.log}")

        logger = logging.getLogger()
        logger.setLevel(numeric_level)

    if args.protocolDir:
        if not os.path.exists(args.protocolDir):
            raise ValueError("Protocol directory does not exist")

        ABFConverter.protocolStorageDir = args.protocolDir

    if args.realDataChannel:
        ABFConverter.adcNamesWithRealData.append(args.realDataChannel)

    for fileOrFolder in args.filesOrFolders:
        print(f"Converting {fileOrFolder}")
        convert(fileOrFolder,
                overwrite=args.overwrite,
                fileType=args.fileType,
                outputMetadata=args.outputMetadata,
                outputFeedbackChannel=args.outputFeedbackChannel,
                multipleGroupsPerFile=args.multipleGroupsPerFile,
                compression=args.compression)


if __name__ == "__main__":
    main()
