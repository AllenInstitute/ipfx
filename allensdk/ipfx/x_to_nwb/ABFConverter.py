from hashlib import sha256
from datetime import datetime
import json
import os
import glob

import numpy as np

import pyabf

from pynwb.device import Device
from pynwb import NWBHDF5IO, NWBFile
from pynwb.icephys import IntracellularElectrode

from allensdk.ipfx.x_to_nwb.utils import PLACEHOLDER, V_CLAMP_MODE, I_CLAMP_MODE, \
     parseUnit, getStimulusSeriesClass, getAcquiredSeriesClass, createSeriesName, createCompressedDataset

ABF_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

# TODO
# add abstract base class


class ABFConverter:

    atfStorage = None

    def __init__(self, inFileOrFolder, outFile):
        """
        Convert the given ABF file to NWB

        Keyword arguments:
        inFileOrFolder -- input file, or folder with multiple files, in ABF v2 format
        outFile        -- target filepath (must not exist)
        """

        inFiles = []

        if os.path.isfile(inFileOrFolder):
            inFiles.append(inFileOrFolder)
        elif os.path.isdir(inFileOrFolder):
            inFiles = glob.glob(os.path.join(inFileOrFolder, "*.abf"))
        else:
            raise ValueError(f"{inFileOrFolder} is neither a folder nor a path.")

        self.abfs = []

        for inFile in inFiles:
            abf = pyabf.ABF(inFile, preLoadData=True, atfStorage=ABFConverter.atfStorage)
            abf.baseline()  # turn off baseline subtraction
            self.abfs.append(abf)

            # ensure that the input file matches our expectations
            self._check(abf)

        self.refabf = self._getOldestABF()

        self._checkAll()

        self.totalSeriesCount = self._getMaxTimeSeriesCount()

        nwbFile = self._createFile()

        device = self._createDevice()
        nwbFile.add_device(device)

        electrodes = self._createElectrodes(device)
        nwbFile.add_ic_electrode(electrodes)

        for i in self._createStimulusSeries(electrodes):
            nwbFile.add_stimulus(i)

        for i in self._createAcquiredSeries(electrodes):
            nwbFile.add_acquisition(i)

        with NWBHDF5IO(outFile, "w") as io:
            io.write(nwbFile, cache_spec=True)

    def _check(self, abf):
        """
        Check that all prerequisites are met.
        """

        if abf.abfFileFormat != 2:
            raise ValueError(f"Can not handle ABF file format version {abf.abfFileFormat} sweeps.")
        elif not (abf.sweepPointCount > 0):
            raise ValueError("The number of data points is not larger than zero.")
        elif not (abf.sweepCount > 0):
            raise ValueError("Found no sweeps.")
        elif not (abf.channelCount > 0):
            raise ValueError("Found no channels.")
        elif sum(abf._dacSection.nWaveformEnable) == 0:
            raise ValueError("All channels are turned off.")
        elif len(np.unique(abf._adcSection.nTelegraphInstrument)) > 1:
            raise ValueError("Unexpected mismatching telegraph instruments.")
        elif len(abf._adcSection.sTelegraphInstrument[0]) == 0:
            raise ValueError("Empty telegraph name.")
        elif len(abf._protocolSection.sDigitizerType) == 0:
            raise ValueError("Empty digitizer type.")
        elif abf.channelCount != len(abf.channelList):
            raise ValueError("Internal channel count is inconsistent.")
        elif abf.sweepCount != len(abf.sweepList):
            raise ValueError("Internal sweep count is inconsistent.")

        for sweep in range(abf.sweepCount):
            for channel in range(abf.channelCount):
                abf.setSweep(sweep, channel=channel)

                if abf.sweepUnitsX != "sec":
                    raise ValueError(f"Unexpected x units of {abf.sweepUnitsX}.")

                if not abf._dacSection.nWaveformEnable[channel]:
                    continue

                if np.isnan(abf.sweepC).any():
                    raise ValueError(f"Found at least one 'Not a Number' "
                                     "entry in stimulus channel {channel} of sweep {sweep}.")

    def _checkAll(self):
        """
        Check that all loaded ABF files have a minimum list of properties in common.

        These are:
        - Digitizer device
        - Telegraph device
        - Creator Name
        - Creator Version
        - abfVersion
        - channelList
        """

        for abf in self.abfs:
            if self.refabf._protocolSection.sDigitizerType != abf._protocolSection.sDigitizerType:
                raise ValueError("Digitizer type does not match.")
            elif self.refabf._adcSection.sTelegraphInstrument[0] != abf._adcSection.sTelegraphInstrument[0]:
                raise ValueError("Telegraph instrument does not match.")
            elif self.refabf._stringsIndexed.uCreatorName != abf._stringsIndexed.uCreatorName:
                raise ValueError("Creator Name does not match.")
            elif self.refabf.creatorVersion != abf.creatorVersion:
                raise ValueError("Creator Version does not match.")
            elif self.refabf.abfVersion != abf.abfVersion:
                raise ValueError("abfVersion does not match.")
            elif self.refabf.channelList != abf.channelList:
                raise ValueError("channelList does not match.")

    def _getOldestABF(self):
        """
        Return the ABF file with the oldest starting time stamp.
        """

        def getTimestamp(abf):
            return abf.abfDateTime

        return min(self.abfs, key=getTimestamp)

    def _getClampMode(self, abf, channel):
        """
        Return the clamp mode of the given channel.
        """

        return abf._adcSection.nTelegraphMode[channel]

    def _getMaxTimeSeriesCount(self):
        """
        Return the maximum number of TimeSeries which will be created from all ABF files.
        """

        def getCount(abf):
            return abf.sweepCount * abf.channelCount

        return sum(map(getCount, self.abfs))

    def _createFile(self):
        """
        Create a pynwb NWBFile object from the ABF file contents.
        """

        def formatVersion(version):
            return f"{version['major']}.{version['minor']}.{version['bugfix']}.{version['build']}"

        def getFileComments(abfs):
            """
            Return the file comments of all files. Returns an empty string if none are present.
            """

            comments = {}

            for abf in abfs:
                if len(abf.abfFileComment) > 0:
                    comments[os.path.basename(abf.abfFilePath)] = abf.abfFileComment

            if not len(comments):
                return ""

            return json.dumps(comments)

        source = PLACEHOLDER
        session_description = getFileComments(self.abfs)
        identifier = sha256(" ".join([abf.fileGUID for abf in self.abfs]).encode()).hexdigest()
        session_start_time = self.refabf.abfDateTime
        creatorName = self.refabf._stringsIndexed.uCreatorName
        creatorVersion = formatVersion(self.refabf.creatorVersion)
        experiment_description = (f"{creatorName} v{creatorVersion}")
        session_id = PLACEHOLDER

        return NWBFile(source=source,
                       session_description=session_description,
                       file_create_date=datetime.utcnow().isoformat(),
                       identifier=identifier,
                       session_start_time=session_start_time,
                       experimenter=None,
                       experiment_description=experiment_description,
                       session_id=session_id)

    def _createDevice(self):
        """
        Create a pynwb Device object from the ABF file contents.
        """

        digitizer = self.refabf._protocolSection.sDigitizerType
        telegraph = self.refabf._adcSection.sTelegraphInstrument[0]

        return Device(f"{digitizer} with {telegraph}", source=PLACEHOLDER)

    def _createElectrodes(self, device):
        """
        Create pynwb ic_electrodes objects from the ABF file contents.
        """

        return [IntracellularElectrode(f"Electrode {x:d}",
                                       device,
                                       source=PLACEHOLDER,
                                       description=PLACEHOLDER)
                for x in self.refabf.channelList]

    def _calculateStartingTime(self, abf):
        """
        Calculate the starting time of the current sweep of `abf` relative to the reference ABF file.
        """

        refTimestamp = datetime.strptime(self.refabf.abfDateTime, ABF_TIMESTAMP_FORMAT)
        timestamp = datetime.strptime(abf.abfDateTime, ABF_TIMESTAMP_FORMAT)

        delta = timestamp - refTimestamp

        return delta.total_seconds() + abf.sweepX[0]

    def _createStimulusSeries(self, electrodes):
        """
        Return a list of pynwb stimulus series objects created from the ABF file contents.
        """

        series = []
        counter = 0

        for abf in self.abfs:
            for sweep in range(abf.sweepCount):
                for channel in range(abf.channelCount):

                    if not abf._dacSection.nWaveformEnable[channel]:
                        continue

                    abf.setSweep(sweep, channel=channel, absoluteTime=True)
                    name, counter = createSeriesName("sweep", counter, total=self.totalSeriesCount)
                    data = createCompressedDataset(abf.sweepC)
                    conversion, unit = parseUnit(abf.sweepUnitsC)
                    electrode = electrodes[channel]
                    gain = abf._dacSection.fDACScaleFactor[channel]
                    resolution = np.nan
                    starting_time = self._calculateStartingTime(abf)
                    rate = float(abf.dataRate)
                    source = PLACEHOLDER
                    description = json.dumps({"protocol": abf.protocol,
                                              "protocolPath": abf.protocolPath,
                                              "name": abf.dacNames[channel],
                                              "number": abf._dacSection.nDACNum[channel]},
                                             sort_keys=True, indent=4)

                    seriesClass = getStimulusSeriesClass(self._getClampMode(abf, channel))

                    stimulus = seriesClass(name=name,
                                           source=source,
                                           data=data,
                                           unit=unit,
                                           electrode=electrode,
                                           gain=gain,
                                           resolution=resolution,
                                           conversion=conversion,
                                           starting_time=starting_time,
                                           rate=rate,
                                           description=description)

                    series.append(stimulus)

        return series

    def _createAcquiredSeries(self, electrodes):
        """
        Return a list of pynwb acquisition series objects created from the ABF file contents.
        """

        series = []
        counter = 0

        for abf in self.abfs:
            for sweep in range(abf.sweepCount):
                for channel in range(abf.channelCount):
                    abf.setSweep(sweep, channel=channel, absoluteTime=True)
                    name, counter = createSeriesName("sweep", counter, total=self.totalSeriesCount)
                    data = createCompressedDataset(abf.sweepY)
                    conversion, unit = parseUnit(abf.sweepUnitsY)
                    electrode = electrodes[channel]
                    gain = abf._adcSection.fADCProgrammableGain[channel]
                    resolution = np.nan
                    starting_time = self._calculateStartingTime(abf)
                    rate = float(abf.dataRate)
                    source = PLACEHOLDER
                    description = json.dumps({"protocol": abf.protocol,
                                              "protocolPath": abf.protocolPath,
                                              "name": abf.adcNames[channel],
                                              "number": abf._adcSection.nADCNum[channel]},
                                             sort_keys=True, indent=4)

                    clampMode = self._getClampMode(abf, channel)
                    seriesClass = getAcquiredSeriesClass(clampMode)

                    if clampMode == V_CLAMP_MODE:
                        capacitance_slow = np.nan
                        capacitance_fast = np.nan
                        resistance_comp_correction = np.nan
                        resistance_comp_bandwidth = np.nan
                        resistance_comp_prediction = np.nan
                        whole_cell_capacitance_comp = np.nan
                        whole_cell_series_resistance_comp = np.nan

                        acquistion_data = seriesClass(name=name,
                                                      source=source,
                                                      data=data,
                                                      unit=unit,
                                                      electrode=electrode,
                                                      gain=gain,
                                                      resolution=resolution,
                                                      conversion=conversion,
                                                      starting_time=starting_time,
                                                      rate=rate,
                                                      description=description,
                                                      capacitance_slow=capacitance_slow,
                                                      capacitance_fast=capacitance_fast,
                                                      resistance_comp_correction=resistance_comp_correction,
                                                      resistance_comp_bandwidth=resistance_comp_bandwidth,
                                                      resistance_comp_prediction=resistance_comp_prediction,
                                                      whole_cell_capacitance_comp=whole_cell_capacitance_comp,
                                                      whole_cell_series_resistance_comp=whole_cell_series_resistance_comp)  # noqa: E501

                    elif clampMode == I_CLAMP_MODE:
                        bias_current = np.nan
                        bridge_balance = np.nan
                        capacitance_compensation = np.nan
                        acquistion_data = seriesClass(name=name,
                                                      source=source,
                                                      data=data,
                                                      unit=unit,
                                                      electrode=electrode,
                                                      gain=gain,
                                                      resolution=resolution,
                                                      conversion=conversion,
                                                      starting_time=starting_time,
                                                      rate=rate,
                                                      description=description,
                                                      bias_current=bias_current,
                                                      bridge_balance=bridge_balance,
                                                      capacitance_compensation=capacitance_compensation)
                    else:
                        raise ValueError(f"Unsupported clamp mode {clampMode}.")

                    series.append(acquistion_data)

        return series
