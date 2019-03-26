## Converting ABF/DAT files to NWB

The script `run_x_to_nwb_conversion.py` allows to convert ABF/DAT files to NeurodataWithoutBorders v2 files.

### ABF specialities

As of 9/2018 PClamp/Clampex does not record all required amplifier settings.
To workaround that issue we've developed `mcc_get_settings.py` which gathers
all amplifier settings from all active amplifiers and writes them to a file in
JSON output.

#### MCC settings gathering

```sh
mcc_get_settings.py --filename 2018_09_12_0003.json --settingsFile misc-settings.json
```

The optional parameter `--filename` gives the name of the output file,
`--settingsFile` is mandatory and makes the connection between the names of the
AD channels and the amplifier names. In addition it holds the optional scale
factors for the stimulus sets.

Example for `misc-settings.json`:

```json
{
    "IN0": "Demo1_1",
    "IN1": "Demo1_2",
    "ScaleFactors": {
        "C1NSD1SHORT": 1.05,
        "C1NSD2SHORT": 1.05,
        "CHIRP": 1,
        "LSFINEST": 1.05,
        "SSFINEST": 7,
        "TRIPPLE": 7
    }
}
```

The JSON files must reside in the same directory as the ABF files.

For continously gathering the amplifier settings when a new ABF file is created
use the `--watchFolder` option.

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

#### Disabling compression

The following command disables compression of the HDF5 datasets (intended for debugging purposes).

```sh
run_x_to_nwb_conversion.py --no-compression 2018_03_20_0000.abf
```

### DAT specialities

#### Required input files

- DAT files acquired with Patchmaster version 2x90.

#### Examples

##### Convert a single file creating one NWB file per Group

```sh
run_x_to_nwb_conversion.py H18.28.015.11.12.dat
```

##### Convert a single file creating one NWB file with all Groups

```sh
run_x_to_nwb_conversion.py --multipleGroupsPerFile H18.28.015.11.12.dat
```

## Creating a PDF from an NWB file for preview purposes

```sh
nwb_to_pdf.py file1.nwb file2.nwb
```

This creates two PDFs named `file1.pdf` and `file2.pdf`.

## Outputting DAT/ABF metadata files for debugging purposes

```sh
run_x_to_nwb_conversion.py --outputMetadata *.dat *.abf
```

## Running the regression tests

Currently only file level regressions tests exist which check that the
conversion from DAT/ABF to NWB results in the same NWB files compared to earlier
versions.

For running the tests do the following:

```sh
cd tests
py.test --collect-only --do-x-nwb-tests test_x_nwb.py
py.test --do-x-nwb-tests --numprocesses auto test_x_nwb.py
```

The separate collection step is necessary as that can not be parallelized, see also
https://github.com/pytest-dev/pytest-xdist/issues/18.
