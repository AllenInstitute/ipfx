"""
Convert ABF files, created by PClamp/Clampex, to NWB v2 files.
"""

from hashlib import sha256
import json
import re
import os
import glob
import warnings
import logging

import numpy as np

import pyabf

from pynwb.device import Device
from pynwb import NWBHDF5IO, NWBFile
from pynwb.icephys import IntracellularElectrode

from ipfx.x_to_nwb.conversion_utils import PLACEHOLDER, V_CLAMP_MODE, I_CLAMP_MODE, I0_CLAMP_MODE, \
     parseUnit, getStimulusSeriesClass, getAcquiredSeriesClass, createSeriesName, convertDataset, \
     getPackageInfo, createCycleID

log = logging.getLogger(__name__)


class ABFConverter:

    protocolStorageDir = None
    adcNamesWithRealData = ["IN 0", "IN 1", "IN 2", "IN 3"]

    def __init__(self, inFileOrFolder, outFile, outputFeedbackChannel, compression=True):
        """
        Convert the given ABF file to NWB

        Keyword arguments:
        inFileOrFolder        -- input file, or folder with multiple files, in ABF v2 format
        outFile               -- target filepath (must not exist)
        outputFeedbackChannel -- Output ADC data from feedback channels as well (useful for debugging only)
        compression           -- Toggle compression for HDF5 datasets
        """

        inFiles = []

        if os.path.isfile(inFileOrFolder):
            inFiles.append(inFileOrFolder)
        elif os.path.isdir(inFileOrFolder):
            inFiles = glob.glob(os.path.join(inFileOrFolder, "*.abf"))
        else:
            raise ValueError(f"{inFileOrFolder} is neither a folder nor a path.")

        self.outputFeedbackChannel = outputFeedbackChannel
        self.compression = compression

        self._settings = self._getJSONFiles(inFileOrFolder)

        self.abfs = []

        for inFile in inFiles:
            abf = pyabf.ABF(inFile, loadData=False, stimulusFileFolder=ABFConverter.protocolStorageDir)
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

    @staticmethod
    def outputMetadata(inFile):
        if not os.path.isfile(inFile):
            raise ValueError(f"The file {inFile} does not exist.")

        root, ext = os.path.splitext(inFile)

        abf = pyabf.ABF(inFile)
        pyabf.abfHeaderDisplay.abfInfoPage(abf).generateHTML(saveAs=root + ".html")

    @staticmethod
    def _getProtocolName(protocolName):
        """
        Return the protocol/stimset name without the channel suffix.
        """

        return re.sub(r"_IN\d+$", "", protocolName)

    def _getJSONFiles(self, inFileOrFolder):
        """
        Search the JSON files with all miscellaneous settings.
        If `inFileOrFolder` is a folder we need one JSON file in that folder or
        multiple JSON files with the same basename as the ABF files.
        If `inFileOrFolder` is a file the JSON file must have the same
        basename.

        Returns a dict with the abf file/folder name as key and a dictinonary with
        the settings as value.
        """

        d = {}

        def loadJSON(filename):
            log.debug(f"Using JSON settings file {filename}.")
            with open(filename) as fh:
                return json.load(fh)

        def addDictEntry(d, filename):
            base, _ = os.path.splitext(filename)
            settings = base + ".json"

            if os.path.isfile(settings):
                d[filename] = loadJSON(settings)
                return None

            warnings.warn(f"Could not find the JSON file {settings} with settings.")

        if os.path.isfile(inFileOrFolder):
            log.debug("Searching JSON files for file conversion.")
            addDictEntry(d, inFileOrFolder)
            return d

        if os.path.isdir(inFileOrFolder):
            files = glob.glob(os.path.join(inFileOrFolder, "*.json"))

            numFiles = len(files)

            log.debug(f"Found {numFiles} JSON files for folder conversion.")

            if numFiles == 0:
                warnings.warn(f"Could not find any JSON file with settings.")
                return d
            elif numFiles == 1:
                # compatibility with old datasets with only one JSON file per folder
                d[inFileOrFolder] = loadJSON(files[0])
                return d

            # iterate over all ABF files
            files = glob.glob(os.path.join(inFileOrFolder, "*.abf"))
            for f in files:
                addDictEntry(d, f)

            return d

    def _check(self, abf):
        """
        Check that all prerequisites are met.
        """

        if abf.abfVersion["major"] != 2:
            raise ValueError(f"Can not handle ABF file format version {abf.abfVersion['major']} sweeps.")
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
                                     f"entry in stimulus channel {channel} of sweep {sweep} "
                                     f"in file {abf.abfFilePath} using protocol {abf.protocol}.")

    def _reduceChannelList(self, abf):
        """
        Return a reduced channel list taking into account the feedback channel export setting.
        """

        if self.outputFeedbackChannel:
            return abf.channelList

        cleanChanneList = []

        for channel in range(abf.channelCount):
            adcName = abf.adcNames[channel]

            if adcName in ABFConverter.adcNamesWithRealData:
                cleanChanneList.append(abf.channelList[channel])

        return cleanChanneList

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
            source = f"({self.refabf.abfFilePath} vs {abf.abfFilePath})"
            if self.refabf._protocolSection.sDigitizerType != abf._protocolSection.sDigitizerType:
                raise ValueError(f"Digitizer type does not match in {source}.")
            elif self.refabf._adcSection.sTelegraphInstrument[0] != abf._adcSection.sTelegraphInstrument[0]:
                raise ValueError(f"Telegraph instrument does not match in {source}.")
            elif self.refabf._stringsIndexed.uCreatorName != abf._stringsIndexed.uCreatorName:
                raise ValueError(f"Creator Name does not match in {source}.")
            elif self.refabf.creatorVersion != abf.creatorVersion:
                raise ValueError(f"Creator Version does not match in {source}.")
            elif self.refabf.abfVersion != abf.abfVersion:
                raise ValueError(f"abfVersion does not match in {source}.")

            refChannelList = self._reduceChannelList(self.refabf)
            channelList = self._reduceChannelList(abf)
            if refChannelList != channelList:
                raise ValueError(f"channelList ({refChannelList} vs {channelList} does not match in {source}.")

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

        session_description = getFileComments(self.abfs)
        if len(session_description) == 0:
            session_description = PLACEHOLDER

        identifier = sha256(" ".join([abf.fileGUID for abf in self.abfs]).encode()).hexdigest()
        session_start_time = self.refabf.abfDateTime
        creatorName = self.refabf._stringsIndexed.uCreatorName
        creatorVersion = formatVersion(self.refabf.creatorVersion)
        experiment_description = (f"{creatorName} v{creatorVersion}")
        source_script_file_name = "run_x_to_nwb_conversion.py"
        source_script = json.dumps(getPackageInfo(), sort_keys=True, indent=4)
        session_id = PLACEHOLDER

        return NWBFile(session_description=session_description,
                       identifier=identifier,
                       session_start_time=session_start_time,
                       experimenter=None,
                       experiment_description=experiment_description,
                       session_id=session_id,
                       source_script_file_name=source_script_file_name,
                       source_script=source_script)

    def _createDevice(self):
        """
        Create a pynwb Device object from the ABF file contents.
        """

        digitizer = self.refabf._protocolSection.sDigitizerType
        telegraph = self.refabf._adcSection.sTelegraphInstrument[0]

        return Device(f"{digitizer} with {telegraph}")

    def _createElectrodes(self, device):
        """
        Create pynwb ic_electrodes objects from the ABF file contents.
        """

        return [IntracellularElectrode(f"Electrode {x:d}",
                                       device,
                                       description=PLACEHOLDER)
                for x in self.refabf.channelList]

    def _calculateStartingTime(self, abf):
        """
        Calculate the starting time of the current sweep of `abf` relative to the reference ABF file.
        """

        delta = abf.abfDateTime - self.refabf.abfDateTime

        return delta.total_seconds() + abf.sweepX[0]

    def _createStimulusSeries(self, electrodes):
        """
        Return a list of pynwb stimulus series objects created from the ABF file contents.
        """

        series = []
        counter = 0

        for file_index, abf in enumerate(self.abfs):

            stimulus_description = ABFConverter._getProtocolName(abf.protocol)
            scale_factor = self._getScaleFactor(abf, stimulus_description)

            for sweep in range(abf.sweepCount):
                cycle_id = createCycleID([file_index, sweep], total=self.totalSeriesCount)
                for channel in range(abf.channelCount):

                    if not abf._dacSection.nWaveformEnable[channel]:
                        continue

                    abf.setSweep(sweep, channel=channel, absoluteTime=True)
                    name, counter = createSeriesName("index", counter, total=self.totalSeriesCount)
                    data = convertDataset(abf.sweepC * scale_factor, self.compression)
                    conversion, unit = parseUnit(abf.sweepUnitsC)
                    electrode = electrodes[channel]
                    gain = abf._dacSection.fDACScaleFactor[channel]
                    resolution = np.nan
                    starting_time = self._calculateStartingTime(abf)
                    rate = float(abf.dataRate)
                    description = json.dumps({"cycle_id": cycle_id,
                                              "protocol": abf.protocol,
                                              "protocolPath": abf.protocolPath,
                                              "file": os.path.basename(abf.abfFilePath),
                                              "name": abf.dacNames[channel],
                                              "number": abf._dacSection.nDACNum[channel]},
                                             sort_keys=True, indent=4)

                    seriesClass = getStimulusSeriesClass(self._getClampMode(abf, channel))

                    if seriesClass is not None:
                        stimulus = seriesClass(name=name,
                                               data=data,
                                               sweep_number=np.uint64(cycle_id),
                                               unit=unit,
                                               electrode=electrode,
                                               gain=gain,
                                               resolution=resolution,
                                               conversion=conversion,
                                               starting_time=starting_time,
                                               rate=rate,
                                               description=description,
                                               stimulus_description=stimulus_description)

                        series.append(stimulus)

        return series

    def _findSettingsEntry(self, abf):
        """
        Return the settings dictionary for the given abf file, either the file
        specific, or the global one for the folder, or None as first tuple element.
        The second element is the source of the data.
        """

        if self._settings is None:
            return None, None

        filename = abf.abfFilePath

        try:
            return self._settings[filename], filename
        except KeyError:
            dirname = os.path.dirname(filename)

            try:
                return self._settings[dirname], dirname
            except KeyError:
                return None, None

    def _getScaleFactor(self, abf, stimset):
        """
        Return the stimulus scale factor for the stimset of the abf file.
        """

        try:
            settings, _ = self._findSettingsEntry(abf)
            return float(settings["ScaleFactors"][stimset])
        except (TypeError, KeyError):
            warnings.warn(f"Could not find the scale factor for the stimset {stimset}, using 1.0 as fallback.")
            return 1.0

    def _getAmplifierSettings(self, abf, clampMode, adcName):
        """
        Return a dict with the amplifier settings read out form the JSON file.
        Unset values are returned as `NaN`.
        """

        d = {}

        try:
            # JSON stores adcName without spaces

            amplifier = "unknown"
            abfSettings, _ = self._findSettingsEntry(abf)
            adcNameWOSpace = adcName.replace(" ", "")
            amplifier = abfSettings["uids"][adcNameWOSpace]
            settings = abfSettings[amplifier]

            if settings["GetMode"] != clampMode:
                warnings.warn(f"Stored clamp mode {settings['GetMode']} does not match requested "
                              f"clamp mode {clampMode} of channel {adcName}.")
                settings = None
        except (TypeError, KeyError):
            warnings.warn(f"Could not find settings for amplifier {amplifier} of channel {adcName}.")
            settings = None

        if settings:
            if clampMode == V_CLAMP_MODE:
                d["capacitance_slow"] = settings["GetSlowCompCap"]
                d["capacitance_fast"] = settings["GetFastCompCap"]

                if settings["GetRsCompEnable"]:
                    d["resistance_comp_correction"] = settings["GetRsCompCorrection"]
                    d["resistance_comp_bandwidth"] = settings["GetRsCompBandwidth"]
                    d["resistance_comp_prediction"] = settings["GetRsCompPrediction"]
                else:
                    d["resistance_comp_correction"] = np.nan
                    d["resistance_comp_bandwidth"] = np.nan
                    d["resistance_comp_prediction"] = np.nan

                if settings["GetWholeCellCompEnable"]:
                    d["whole_cell_capacitance_comp"] = settings["GetWholeCellCompCap"]
                    d["whole_cell_series_resistance_comp"] = settings["GetWholeCellCompResist"]
                else:
                    d["whole_cell_capacitance_comp"] = np.nan
                    d["whole_cell_series_resistance_comp"] = np.nan

            elif clampMode in (I_CLAMP_MODE, I0_CLAMP_MODE):
                if settings["GetHoldingEnable"]:
                    d["bias_current"] = settings["GetHolding"]
                else:
                    d["bias_current"] = np.nan

                if settings["GetBridgeBalEnable"]:
                    d["bridge_balance"] = settings["GetBridgeBalResist"]
                else:
                    d["bridge_balance"] = np.nan

                if settings["GetNeutralizationEnable"]:
                    d["capacitance_compensation"] = settings["GetNeutralizationCap"]
                else:
                    d["capacitance_compensation"] = np.nan
            else:
                warnings.warn("Unsupported clamp mode {clampMode}")
        else:
            if clampMode == V_CLAMP_MODE:
                d["capacitance_slow"] = np.nan
                d["capacitance_fast"] = np.nan
                d["resistance_comp_correction"] = np.nan
                d["resistance_comp_bandwidth"] = np.nan
                d["resistance_comp_prediction"] = np.nan
                d["whole_cell_capacitance_comp"] = np.nan
                d["whole_cell_series_resistance_comp"] = np.nan
            elif clampMode in (I_CLAMP_MODE, I0_CLAMP_MODE):
                d["bias_current"] = np.nan
                d["bridge_balance"] = np.nan
                d["capacitance_compensation"] = np.nan
            else:
                warnings.warn("Unsupported clamp mode {clampMode}")

        return d

    def _createAcquiredSeries(self, electrodes):
        """
        Return a list of pynwb acquisition series objects created from the ABF file contents.
        """

        series = []
        counter = 0

        for file_index, abf in enumerate(self.abfs):

            stimulus_description = ABFConverter._getProtocolName(abf.protocol)
            _, jsonSource = self._findSettingsEntry(abf)
            log.debug(f"Using JSON settings for {jsonSource}.")

            for sweep in range(abf.sweepCount):
                cycle_id = createCycleID([file_index, sweep], total=self.totalSeriesCount)
                for channel in range(abf.channelCount):
                    adcName = abf.adcNames[channel]

                    if not self.outputFeedbackChannel:
                        if adcName in ABFConverter.adcNamesWithRealData:
                            pass
                        else:
                            # feedback data, skip
                            continue

                    abf.setSweep(sweep, channel=channel, absoluteTime=True)
                    name, counter = createSeriesName("index", counter, total=self.totalSeriesCount)
                    data = convertDataset(abf.sweepY, self.compression)
                    conversion, unit = parseUnit(abf.sweepUnitsY)
                    electrode = electrodes[channel]
                    gain = abf._adcSection.fADCProgrammableGain[channel]
                    resolution = np.nan
                    starting_time = self._calculateStartingTime(abf)
                    rate = float(abf.dataRate)
                    description = json.dumps({"cycle_id": cycle_id,
                                              "protocol": abf.protocol,
                                              "protocolPath": abf.protocolPath,
                                              "file": os.path.basename(abf.abfFilePath),
                                              "name": adcName,
                                              "number": abf._adcSection.nADCNum[channel]},
                                             sort_keys=True, indent=4)

                    clampMode = self._getClampMode(abf, channel)
                    settings = self._getAmplifierSettings(abf, clampMode, adcName)
                    seriesClass = getAcquiredSeriesClass(clampMode)

                    if clampMode == V_CLAMP_MODE:
                        acquistion_data = seriesClass(name=name,
                                                      data=data,
                                                      sweep_number=np.uint64(cycle_id),
                                                      unit=unit,
                                                      electrode=electrode,
                                                      gain=gain,
                                                      resolution=resolution,
                                                      conversion=conversion,
                                                      starting_time=starting_time,
                                                      rate=rate,
                                                      description=description,
                                                      capacitance_slow=settings["capacitance_slow"],
                                                      capacitance_fast=settings["capacitance_fast"],
                                                      resistance_comp_correction=settings["resistance_comp_correction"],
                                                      resistance_comp_bandwidth=settings["resistance_comp_bandwidth"],
                                                      resistance_comp_prediction=settings["resistance_comp_prediction"],
                                                      stimulus_description=stimulus_description,
                                                      whole_cell_capacitance_comp=settings["whole_cell_capacitance_comp"],  # noqa: E501
                                                      whole_cell_series_resistance_comp=settings["whole_cell_series_resistance_comp"])  # noqa: E501

                    elif clampMode in (I_CLAMP_MODE, I0_CLAMP_MODE):
                        acquistion_data = seriesClass(name=name,
                                                      data=data,
                                                      sweep_number=np.uint64(cycle_id),
                                                      unit=unit,
                                                      electrode=electrode,
                                                      gain=gain,
                                                      resolution=resolution,
                                                      conversion=conversion,
                                                      starting_time=starting_time,
                                                      rate=rate,
                                                      description=description,
                                                      bias_current=settings["bias_current"],
                                                      bridge_balance=settings["bridge_balance"],
                                                      stimulus_description=stimulus_description,
                                                      capacitance_compensation=settings["capacitance_compensation"])
                    else:
                        raise ValueError(f"Unsupported clamp mode {clampMode}.")

                    series.append(acquistion_data)

        return series
