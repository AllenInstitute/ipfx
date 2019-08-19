#!/bin/env python

import os
import argschema as args
from ipfx.x_to_nwb.NWBConverter import NWBConverter


class ConvertNWBParameters(args.ArgSchema):
    input_nwb_file = args.fields.InputFile(description="input nwb1 file", required=True)


def make_nwb2_file_name(dir_name,base_name):

    file_name, file_extension = os.path.splitext(base_name)
    nwb2_file_name = os.path.join(dir_name, file_name+"_ver2" + file_extension)

    return nwb2_file_name


def main():

    module = args.ArgSchemaParser(schema_type=ConvertNWBParameters)

    nwb1_file_name = module.args["input_nwb_file"]
    dir_name = os.path.dirname(nwb1_file_name)
    base_name = os.path.basename(nwb1_file_name)

    nwb2_file_name = make_nwb2_file_name(dir_name,base_name)

    if not os.path.exists(nwb1_file_name):
        raise ValueError(f"The file {nwb1_file_name} does not exist.")

    NWBConverter(nwb1_file_name,
                 nwb2_file_name,
                 )


if __name__ == "__main__":
    main()
