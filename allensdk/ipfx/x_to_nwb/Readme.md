### Converting ABF files to NWB

The script `run_x_to_nwb_conversion.py` allows to convert ABF files to NeurodataWithoutBorders v2 files.

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
run_x_to_nwb_conversion.py --fileType ".abf" --overwrite someFolder 2018_03_20_0000.abf
```
