## Converting ABF/DAT files to NWB

The script `run_x_to_nwb_conversion.py` allows to convert ABF/DAT files to NeurodataWithoutBorders v2 files.

NWB currently does not offer support for grouping different TimeSeries'
together. We work around that by setting the string "cycle_id" in the description
of the TimeSeries to something unique for TimeSeries' which belong together.
Software reading NWB file can use that information to group the TimeSeries back
together. The string "cycle_id" should be treated as opaque and is subject to
change at any time.

### ABF specialities

As of 9/2018 PClamp/Clampex does not record all required amplifier settings.
To workaround that issue we've developed `mcc_get_settings.py` which gathers
all amplifier settings from all active amplifiers and writes them to a file in
JSON output. This file can then, in a future `run_x_to_nwb_conversion.py`
version, be used to provide the missing amplifier settings.

#### Required input files

- ABF files acquired with Clampex/pCLAMP.
- If custom waveforms are used for the stimulus protocol, the source ATF files are required as well.

#### Examples

##### Convert a single file

```sh
run_x_to_nwb_conversion.py 2018_03_20_0000.abf
```

##### Convert a single file with overwrite and use a directory for finding custom waveforms

Some acquired data might use custom wave forms for defining the stimulus
protocols. These custom waveforms are stored in external files and don't reside
in the ABF files. We therefore allow the user to pass a directory where
these files should be searched. Currently only custom waveforms in ATF (Axon
Text format) are supported.

```sh
run_x_to_nwb_conversion.py --overwrite --protocolDir protocols 2018_03_20_0000.abf
```

##### Convert a folder with ABF files

The following command converts all ABF files which reside in `someFolder` to a single NWB file.

```sh
run_x_to_nwb_conversion.py --fileType ".abf" --overwrite someFolder
```

### DAT specialities

#### Required input files

- DAT files acquired with Patchmaster version 2x90.

#### Examples

##### Convert a single file

```sh
run_x_to_nwb_conversion.py H18.28.015.11.12.dat
```

## Creating a PDF from an NWB file for preview purposes

```sh
nwb_to_pdf.py file1.nwb file2.nwb
```

This creates two PDFs named `file1.pdf` and `file2.pdf`.
