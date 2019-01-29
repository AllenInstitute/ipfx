"""
All supported nodes are listed here. The root node of each bundle calls the
TreeNode constructor explicitly the other are plain children of the root nodes.

Documentation:
    * https://github.com/neurodroid/stimfit/blob/master/src/libstfio/heka/hekalib.cpp
    * ftp://server.hekahome.de/pub/FileFormat/Patchmasterv9/
    * For the field_info type parameters see
      https://docs.python.org/3/library/struct.html#format-characters
    * The CARD types are unsigned, see e.g.
      https://www.common-lisp.net/project/cmucl/doc/clx/1_6_Data_Types.html

The nodes are tailored for patchmaster version 2x90.x.
"""

import numpy as np

from ipfx.x_to_nwb.hr_treenode import TreeNode
from ipfx.x_to_nwb.hr_struct import Struct


def cstr(byte):
    """Convert C string bytes to python string.
    """
    try:
        ind = byte.index(b'\0')
    except ValueError:
        print("Could not find a trailing '\\0'!")
        return byte
    return byte[:ind].decode('utf-8', errors='ignore')


def getFromList(lst, index):

    try:
        return lst[index]
    except IndexError:
        return f"Unknown (value: {index})"


def getAmplifierType(byte):
    return getFromList(["EPC7", "EPC8", "EPC9", "EPC10", "EPC10Plus"], byte)


def getADBoard(byte):
    return getFromList(["ITC16", "ITC18", "LIH1600"], byte)


def getRecordingMode(byte):
    return getFromList(["InOut", "OnCell", "OutOut", "WholeCell", "CClamp", "VClamp", "NoMode"], byte)


def getDataFormat(byte):
    return getFromList(["int16", "int32", "real32", "real64"], byte)


def getSegmentClass(byte):
    return getFromList(["Constant", "Ramp", "Continuous", "ConstSine", "Squarewave", "Chirpwave"], byte)


def getStoreType(byte):
    return getFromList(["NoStore", "Store", "StoreStart", "StoreEnd"], byte)


def getIncrementMode(byte):
    return getFromList(["Inc", "Dec", "IncInterleaved", "DecInterleaved",
                        "Alternate", "LogInc", "LogDec", "LogIncInterleaved",
                        "LogDecInterleaved", "LogAlternate"], byte)


def getSourceType(byte):
    return getFromList(["Constant", "Hold", "Parameter"], byte)


def getAmplifierGain(byte):
    """
    Units: V/A
    """

    # Original units: mV/pA
    return getFromList([1e-3/1e-12 * x for x in
                       [0.005, 0.010, 0.020, 0.050, 0.1, 0.2,
                        0.5, 1, 2, 5, 10, 20,
                        50, 100, 200, 500, 1000, 2000]], byte)


def convertDataFormatToNP(dataFormat):

    d = {"int16": np.int16,
         "int32": np.int32,
         "real32": np.float16,
         "real64": np.float32}

    return d[dataFormat]


def getClampMode(byte):
    return getFromList(["TestMode", "VCMode", "CCMode", "NoMode"], byte)


def getAmplMode(byte):
    return getFromList(["Any", "VCMode", "CCMode", "IDensityMode"], byte)


def getADCMode(byte):
    return getFromList(["AdcOff", "Analog", "Digitals", "Digital", "AdcVirtual"], byte)


def convertDataKind(byte):

    d = {}

    # LittleEndianBit = 0;
    # IsLeak          = 1;
    # IsVirtual       = 2;
    # IsImon          = 3;
    # IsVmon          = 4;
    # Clip            = 5;
    # (*
    #  -> meaning of bits:
    #     - LittleEndianBit => byte sequence
    #       "PowerPC Mac" = cleared
    #       "Windows and Intel Mac" = set
    #     - IsLeak
    #       set if trace is a leak trace
    #     - IsVirtual
    #       set if trace is a virtual trace
    #     - IsImon
    #       -> set if trace was from Imon ADC
    #       -> it flags a trace to be used to
    #          compute LockIn traces from
    #       -> limited to "main" traces, not "leaks"!
    #     - IsVmon
    #       -> set if trace was from Vmon ADC
    #     - Clip
    #       -> set if amplifier of trace was clipping
    # *)

    d["IsLittleEndian"] = bool(byte & (1 << 0))
    d["IsLeak"] = bool(byte & (1 << 1))
    d["IsVirtual"] = bool(byte & (1 << 2))
    d["IsImon"] = bool(byte & (1 << 3))
    d["IsVmon"] = bool(byte & (1 << 4))
    d["Clip"] = bool(byte & (1 << 5))

    return d


def convertStimToDacID(byte):

    d = {}

    # StimToDacID :
    #   Specifies how to convert the Segment
    #   "Voltage" to the actual voltage sent to the DAC
    #   -> meaning of bits:
    #      bit 0 (UseStimScale)    -> use StimScale
    #      bit 1 (UseRelative)     -> relative to Vmemb
    #      bit 2 (UseFileTemplate) -> use file template
    #      bit 3 (UseForLockIn)    -> use for LockIn computation
    #      bit 4 (UseForWavelength)
    #      bit 5 (UseScaling)
    #      bit 6 (UseForChirp)
    #      bit 7 (UseForImaging)
    #      bit 14 (UseReserved)
    #      bit 15 (UseReserved)

    d["UseStimScale"] = bool(byte & (1 << 0))
    d["UseRelative"] = bool(byte & (1 << 1))
    d["UseFileTemplate"] = bool(byte & (1 << 2))
    d["UseForLockIn"] = bool(byte & (1 << 3))
    d["UseForWavelength"] = bool(byte & (1 << 4))
    d["UseScaling"] = bool(byte & (1 << 5))
    d["UseForChirp"] = bool(byte & (1 << 6))
    d["UseForImaging"] = bool(byte & (1 << 7))

    return d


def getSquareKind(byte):
    return getFromList(["Common Frequency"], byte)


def getChirpKind(byte):
    return getFromList(["Linear", "Exponential", "Spectroscopic"], byte)


class Marker(TreeNode):

    field_info = [
        ('Version', 'i'),  # (* INT32 *)
        ('CRC', 'I'),      # (* CARD32 *)
    ]

    required_size = 8

    rectypes = [
        None
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class Solutions(TreeNode):

    field_info = [
        ('RoVersion', 'H'),               # (* INT16 *)
        ('RoDataBaseName', '80s', cstr),  # (* SolutionNameSize *)
        ('RoSpare1', 'H', None),          # (* INT16 *)
        ('RoCRC', 'I'),                   # (* CARD32 *)
    ]

    required_size = 88

    rectypes = [
        None
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class ProtocolMethod(TreeNode):

    field_info = [
        ('Version', 'i'),              # (* INT32 *)
        ('Mark', 'i'),                 # (* INT32 *)
        ('VersionName', '32s', cstr),  # (* String32Type *)
        ('MaxSamples', 'i'),           # (* INT32 *)
        ('Filler1', 'i', None),        # (* INT32 *)
        ('Params', '10s', cstr),       # (* ARRAY[0..9] OF LONGREAL
                                       # ('StimParams', ''),    *)
        ('ParamText', '320s', cstr),   # (* ARRAY[0..9],[0..31]OF CHAR
                                       # ('StimParamChars', '') *)
        ('Reserved', '128s', None),    # (* String128Type *)
        ('Filler2', 'i', None),        # (* INT32 *)
        ('CRC', 'I'),                  # (* CARD32 *)
    ]

    required_size = 514

    rectypes = [
        None
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class LockInParams(Struct):
    field_info = [
        ('ExtCalPhase', 'd'),      # (* LONGREAL *)
        ('ExtCalAtten', 'd'),      # (* LONGREAL *)
        ('PLPhase', 'd'),          # (* LONGREAL *)
        ('PLPhaseY1', 'd'),        # (* LONGREAL *)
        ('PLPhaseY2', 'd'),        # (* LONGREAL *)
        ('UsedPhaseShift', 'd'),   # (* LONGREAL *)
        ('UsedAttenuation', 'd'),  # (* LONGREAL *)
        ('Spares2', 'd', None),    # (* LONGREAL *)
        ('ExtCalValid', '?'),      # (* BOOLEAN *)
        ('PLPhaseValid', '?'),     # (* BOOLEAN *)
        ('LockInMode', 'b'),       # (* BYTE *)
        ('CalMode', 'b'),          # (* BYTE *)
        ('Spares', '28s', None),   # (* remaining *)
    ]

    required_size = 96


class AmplifierState(Struct):
    field_info = [
        ('StateVersion', '8s', cstr),         # (* 8 = SizeStateVersion *)
        ('RealCurrentGain', 'd'),             # (* LONGREAL *)
        ('RealF2Bandwidth', 'd'),             # (* LONGREAL *)
        ('F2Frequency', 'd'),                 # (* LONGREAL *)
        ('RsValue', 'd'),                     # (* LONGREAL *)
        ('RsFraction', 'd'),                  # (* LONGREAL *)
        ('GLeak', 'd'),                       # (* LONGREAL *)
        ('CFastAmp1', 'd'),                   # (* LONGREAL *)
        ('CFastAmp2', 'd'),                   # (* LONGREAL *)
        ('CFastTau', 'd'),                    # (* LONGREAL *)
        ('CSlow', 'd'),                       # (* LONGREAL *)
        ('GSeries', 'd'),                     # (* LONGREAL *)
        ('StimDacScale', 'd'),                # (* LONGREAL *)
        ('CCStimScale', 'd'),                 # (* LONGREAL *)
        ('VHold', 'd'),                       # (* LONGREAL *)
        ('LastVHold', 'd'),                   # (* LONGREAL *)
        ('VpOffset', 'd'),                    # (* LONGREAL *)
        ('VLiquidJunction', 'd'),             # (* LONGREAL *)
        ('CCIHold', 'd'),                     # (* LONGREAL *)
        ('CSlowStimVolts', 'd'),              # (* LONGREAL *)
        ('CCTrackVHold', 'd'),                # (* LONGREAL *)
        ('TimeoutLength', 'd'),               # (* LONGREAL *)
        ('SearchDelay', 'd'),                 # (* LONGREAL *)
        ('MConductance', 'd'),                # (* LONGREAL *)
        ('MCapacitance', 'd'),                # (* LONGREAL *)
        ('SerialNumber', '8s', cstr),         # (* 8 = SizeSerialNumber *)
        ('E9Boards', 'h'),                    # (* INT16 *)
        ('CSlowCycles', 'h'),                 # (* INT16 *)
        ('IMonAdc', 'h'),                     # (* INT16 *)
        ('VMonAdc', 'h'),                     # (* INT16 *)
        ('MuxAdc', 'h'),                      # (* INT16 *)
        ('TstDac', 'h'),                      # (* INT16 *)
        ('StimDac', 'h'),                     # (* INT16 *)
        ('StimDacOffset', 'h'),               # (* INT16 *)
        ('MaxDigitalBit', 'h'),               # (* INT16 *)
        ('HasCFastHigh', 'b'),                # (* BYTE *)
        ('CFastHigh', 'b'),                   # (* BYTE *)
        ('HasBathSense', 'b'),                # (* BYTE *)
        ('BathSense', 'b'),                   # (* BYTE *)
        ('HasF2Bypass', 'b'),                 # (* BYTE *)
        ('F2Mode', 'b'),                      # (* BYTE *)
        ('AmplKind', 'b', getAmplifierType),  # (* BYTE *)
        ('IsEpc9N', 'b'),                     # (* BYTE *)
        ('ADBoard', 'b', getADBoard),         # (* BYTE *)
        ('BoardVersion', 'b'),                # (* BYTE *)
        ('ActiveE9Board', 'b'),               # (* BYTE *)
        ('Mode', 'b', getClampMode),          # (* BYTE *)
                                              # Modes = (TestMode, VCMode,
                                              #          CCMode, NoMode => (* AmplifierState is invalid *));
        ('Range', 'b'),                       # (* BYTE *)
        ('F2Response', 'b'),                  # (* BYTE *)
        ('RsOn', 'b'),                        # (* BYTE *)
        ('CSlowRange', 'b'),                  # (* BYTE *)
        ('CCRange', 'b'),                     # (* BYTE *)
        ('CCGain', 'b'),                      # (* BYTE *)
        ('CSlowToTstDac', 'b'),               # (* BYTE *)
        ('StimPath', 'b'),                    # (* BYTE *)
        ('CCTrackTau', 'b'),                  # (* BYTE *)
        ('WasClipping', 'b'),                 # (* BYTE *)
        ('RepetitiveCSlow', 'b'),             # (* BYTE *)
        ('LastCSlowRange', 'b'),              # (* BYTE *)
        ('Old2', 'b', None),                  # (* BYTE *)
        ('CanCCFast', 'b'),                   # (* BYTE *)
        ('CanLowCCRange', 'b'),               # (* BYTE *)
        ('CanHighCCRange', 'b'),              # (* BYTE *)
        ('CanCCTracking', 'b'),               # (* BYTE *)
        ('HasVmonPath', 'b'),                 # (* BYTE *)
        ('HasNewCCMode', 'b'),                # (* BYTE *)
        ('Selector', 'c'),                    # (* CHAR *)
        ('HoldInverted', 'b'),                # (* BYTE *)
        ('AutoCFast', '?'),                   # (* BYTE *)
        ('AutoCSlow', '?'),                   # (* CHAR *)
        ('HasVmonX100', 'b'),                 # (* BYTE *)
        ('TestDacOn', 'b'),                   # (* BYTE *)
        ('QMuxAdcOn', 'b'),                   # (* BYTE *)
        ('Imon1Bandwidth', 'd'),              # (* LONGREAL *)
        ('StimScale', 'd'),                   # (* LONGREAL *)
        ('Gain', 'b', getAmplifierGain),      # (* BYTE *)
        ('Filter1', 'b'),                     # (* BYTE *)
        ('StimFilterOn', 'b'),                # (* BYTE *)
        ('RsSlow', 'b'),                      # (* BYTE *)
        ('Old1', 'b', None),                  # (* BYTE *)
        ('CCCFastOn', '?'),                   # (* BYTE *)
        ('CCFastSpeed', 'b'),                 # (* BYTE *)
        ('F2Source', 'b'),                    # (* BYTE *)
        ('TestRange', 'b'),                   # (* BYTE *)
        ('TestDacPath', 'b'),                 # (* BYTE *)
        ('MuxChannel', 'b'),                  # (* BYTE *)
        ('MuxGain64', 'b'),                   # (* BYTE *)
        ('VmonX100', 'b'),                    # (* BYTE *)
        ('IsQuadro', 'b'),                    # (* BYTE *)
        ('F1Mode', 'b'),                      # (* BYTE *)
        ('Old3', 'b', None),                  # (* BYTE *)
        ('StimFilterHz', 'd'),                # (* LONGREAL *)
        ('RsTau', 'd'),                       # (* LONGREAL *)
        ('DacToAdcDelay', 'd'),               # (* LONGREAL *)
        ('InputFilterTau', 'd'),              # (* LONGREAL *)
        ('OutputFilterTau', 'd'),             # (* LONGREAL *)
        ('vMonFactor', 'd', None),            # (* LONGREAL *)
        ('CalibDate', '16s', cstr),           # (* 16 = SizeCalibDate *)
        ('VmonOffset', 'd'),                  # (* LONGREAL *)
        ('EEPROMKind', 'b'),                  # (* BYTE *)
        ('VrefX2', 'b'),                      # (* BYTE *)
        ('HasVrefX2AndF2Vmon', 'b'),          # (* BYTE *)
        ('Spare1', 'b', None),                # (* BYTE *)
        ('Spare2', 'b', None),                # (* BYTE *)
        ('Spare3', 'b', None),                # (* BYTE *)
        ('Spare4', 'b', None),                # (* BYTE *)
        ('Spare5', 'b', None),                # (* BYTE *)
        ('CCStimDacScale', 'd'),              # (* LONGREAL *)
        ('VmonFiltBandwidth', 'd'),           # (* LONGREAL *)
        ('VmonFiltFrequency', 'd'),           # (* LONGREAL *)
    ]

    required_size = 400


class AmplifierStateRecord(TreeNode):

    field_info = [
       ('Mark', 'i'),                       # (* INT32 *)
       ('StateCount', 'i'),                 # (* INT32 *)
       ('StateVersion', 'b'),               # (* CHAR *)
       ('Filler1', 'b', None),              # (* BYTE *)
       ('Filler2', 'b', None),              # (* BYTE *)
       ('Filler3', 'b', None),              # (* BYTE *)
       ('Filler4', 'i', None),              # (* INT32 *)
       ('LockInParams', LockInParams),      # (* LockInParamsSize'   , ' ') , *)
       ('AmplifierState', AmplifierState),  # (* AmplifierStateSize', ' ')  , *)
       ('IntSol', 'i'),                     # (* INT32 *)
       ('ExtSol', 'i'),                     # (* INT32 *)
       ('Filler5', '36s', None),            # (* spares: bytes *)
       ('CRC', 'I')                         # (* CARD32 *)
    ]

    required_size = 560


class AmplifierSeriesRecord(TreeNode):

    field_info = [
       ('Mark', 'i'),           # (* INT32 *)
       ('SeriesCount', 'i'),    # (* INT32 *)
       ('Filler1', 'i', None),  # (* INT32 *)
       ('CRC', 'I'),            # (* CARD32 *)
    ]

    required_size = 16


class AmplifierFile(TreeNode):

    field_info = [
        ('Version', 'i'),                # (* INT32 *)
        ('Mark', 'i'),                   # (* INT32 *)
        ('VersionName', '32s', cstr),    # (* String32Type *)
        ('AmplifierName', '32s', cstr),  # (* String32Type *)
        ('Amplifier', 'c'),              # (* CHAR *)
        ('ADBoard', 'c'),                # (* CHAR *)
        ('Creator', 'c'),                # (* CHAR *)
        ('Filler1', 'b', None),          # (* BYTE *)
        ('CRC', 'I')                     # (* CARD32 *)
    ]

    required_size = 80

    rectypes = [
        None,
        AmplifierSeriesRecord,
        AmplifierStateRecord
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class UserParamDescrType(Struct):
    field_info = [
        ('Name', '32s', cstr),
        ('Unit', '8s', cstr),
    ]

    required_size = 40


class GroupRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),              # (* INT32 *)
        ('Label', '32s', cstr),     # (* String32Size *)
        ('Text', '80s', cstr),      # (* String80Size *)
        ('ExperimentNumber', 'i'),  # (* INT32 *)
        ('GroupCount', 'i'),        # (* INT32 *)
        ('CRC', 'I'),               # (* CARD32 *)
        ('MatrixWidth', 'd'),       # (* LONGREAL *)
        ('MatrixHeight', 'd'),      # (* LONGREAL *)
    ]

    required_size = 144


class SeriesRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),                                       # (* INT32 *)
        ('Label', '32s', cstr),                              # (* String32Type *)
        ('Comment', '80s', cstr),                            # (* String80Type *)
        ('SeriesCount', 'i'),                                # (* INT32 *)
        ('NumberSweeps', 'i'),                               # (* INT32 *)
        ('AmplStateOffset', 'i'),                            # (* INT32 *)
        ('AmplStateSeries', 'i'),                            # (* INT32 *)
        ('MethodTag', 'i'),                                  # (* INT32 *)
        ('Time', 'd'),                                       # (* LONGREAL *)
        ('PageWidth', 'd'),                                  # (* LONGREAL *)
        ('SwUserParamDescr', UserParamDescrType.array(4)),   # (* ARRAY[0..3] OF UserParamDescrType = 4*40 *)
        ('MethodName', '32s', None),                         # (* String32Type *)
        ('UserParams', '4d'),                                # (* ARRAY[0..3] OF LONGREAL *)
        ('LockInParams', LockInParams),                      # (* SeLockInSize = 96, see "Pulsed.de" *)
        ('AmplifierState', AmplifierState),                  # (* AmplifierStateSize = 400 *)
        ('Username', '80s', cstr),                           # (* String80Type *)
        ('SeUserParamDescr1', UserParamDescrType.array(4)),  # (* ARRAY[0..3] OF UserParamDescrType = 4*40 *)
        ('Filler1', 'i', None),                              # (* INT32 *)
        ('CRC', 'I'),                                        # (* CARD32 *)
        ('SeUserParams2', '4d'),                             # (* ARRAY[0..3] OF LONGREAL *)
        ('SeUserParamDescr2', UserParamDescrType.array(4)),  # (* ARRAY[0..3] OF UserParamDescrType = 4*40 *)
        ('ScanParams', '96s', cstr),                         # (* ScanParamsSize = 96 *)
    ]

    required_size = 1408


class SweepRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),               # (* INT32 *)
        ('Label', '32s', cstr),      # (* String32Type *)
        ('AuxDataFileOffset', 'i'),  # (* INT32 *)
        ('StimCount', 'i'),          # (* INT32 *)
        ('SweepCount', 'i'),         # (* INT32 *)
        ('Time', 'd'),               # (* LONGREAL *)
        ('Timer', 'd'),              # (* LONGREAL *)
        ('SwUserParams', '4d'),      # (* ARRAY[0..3] OF LONGREAL *)
        ('Temperature', 'd'),        # (* LONGREAL *)
        ('OldIntSol', 'i'),          # (* INT32 *)
        ('OldExtSol', 'i'),          # (* INT32 *)
        ('DigitalIn', 'h'),          # (* SET16 *)
        ('SweepKind', 'h'),          # (* SET16 *)
        ('DigitalOut', 'h'),         # (* SET16 *)
        ('Filler1', 'h', None),      # (* INT16 *)
        ('Markers', '4d'),           # (* ARRAY[0..3] OF LONGREAL, see SwMarkersNo *)
        ('Filler2', 'i', None),      # (* INT32 *)                                    ),
        ('CRC', 'I'),                # (* CARD32 *)
        ('SwHolding', '16d')         # (* ARRAY[0..15] OF LONGREAL, see SwHoldingNo *)
    ]

    required_size = 288


class TraceRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),                  # (* INT32 *)
        ('Label', '32s', cstr),         # (* String32Type *)
        ('TraceCount', 'i'),            # (* INT32 *)
        ('Data', 'i'),                  # (* INT32 *)
        ('DataPoints', 'i'),            # (* INT32 *)
        ('InternalSolution', 'i'),      # (* INT32 *)
        ('AverageCount', 'i'),          # (* INT32 *)
        ('LeakCount', 'i'),             # (* INT32 *)
        ('LeakTraces', 'i'),            # (* INT32 *)
        ('DataKind', 'h', convertDataKind),  # (* SET16 *)
        ('UseXStart', '?'),             # (* BOOLEAN *)
        ('Kind', 'b'),                  # (* BYTE *)
        ('RecordingMode', 'b', getRecordingMode),  # (* BYTE *)
        ('AmplIndex', 'b'),             # (* BYTE *)
        ('DataFormat', 'b', getDataFormat),  # (* CHAR *)
        ('DataAbscissa', 'b'),          # (* BYTE *)
        ('DataScaler', 'd'),            # (* LONGREAL *)
        ('TimeOffset', 'd'),            # (* LONGREAL *)
        ('ZeroData', 'd'),              # (* LONGREAL *)
        ('YUnit', '8s', cstr),          # (* String8Type *)
        ('XInterval', 'd'),             # (* LONGREAL *)
        ('XStart', 'd'),                # (* LONGREAL *)
        ('XUnit', '8s', cstr),          # (* String8Type *)
        ('YRange', 'd'),                # (* LONGREAL *)
        ('YOffset', 'd'),               # (* LONGREAL *)
        ('Bandwidth', 'd'),             # (* LONGREAL *)
        ('PipetteResistance', 'd'),     # (* LONGREAL *)
        ('CellPotential', 'd'),         # (* LONGREAL *)
        ('SealResistance', 'd'),        # (* LONGREAL *)
        ('CSlow', 'd'),                 # (* LONGREAL *)
        ('GSeries', 'd'),               # (* LONGREAL *)
        ('RsValue', 'd'),               # (* LONGREAL *)
        ('GLeak', 'd'),                 # (* LONGREAL *)
        ('MConductance', 'd'),          # (* LONGREAL *)
        ('LinkDAChannel', 'i'),         # (* INT32 *)
        ('ValidYrange', '?'),           # (* BOOLEAN *)
        ('AdcMode', 'b', getADCMode),   # (* CHAR *)
        ('AdcChannel', 'h'),            # (* INT16 *)
        ('Ymin', 'd'),                  # (* LONGREAL *)
        ('Ymax', 'd'),                  # (* LONGREAL *)
        ('SourceChannel', 'i'),         # (* INT32 *)
        ('ExternalSolution', 'i'),      # (* INT32 *)
        ('CM', 'd'),                    # (* LONGREAL *)
        ('GM', 'd'),                    # (* LONGREAL *)
        ('Phase', 'd'),                 # (* LONGREAL *)
        ('DataCRC', 'I'),               # (* CARD32 *)
        ('CRC', 'I'),                   # (* CARD32 *)
        ('GS', 'd'),                    # (* LONGREAL *)
        ('SelfChannel', 'i'),           # (* INT32 *)
        ('InterleaveSize', 'i'),        # (* INT32 *)
        ('InterleaveSkip', 'i'),        # (* INT32 *)
        ('ImageIndex', 'i'),            # (* INT32 *)
        ('Markers', '10d'),             # (* ARRAY[0..9] OF LONGREAL *)
        ('SECM_X', 'd'),                # (* LONGREAL *)
        ('SECM_Y', 'd'),                # (* LONGREAL *)
        ('SECM_Z', 'd'),                # (* LONGREAL *)
        ('Holding', 'd'),               # (* LONGREAL *)
        ('Enumerator', 'i'),            # (* INT32 *)
        ('XTrace', 'i'),                # (* INT32 *)
        ('IntSolValue', 'd'),           # (* LONGREAL *)
        ('ExtSolValue', 'd'),           # (* LONGREAL *)
        ('IntSolName', '32s', cstr),    # (* String32Size *)
        ('ExtSolName', '32s', cstr),    # (* String32Size *)
        ('DataPedestal', 'd'),          # (* LONGREAL *)
    ]

    required_size = 512


class Pulsed(TreeNode):
    field_info = [
        ('Version', 'i'),              # (* INT32 *)
        ('Mark', 'i'),                 # (* INT32 *)
        ('VersionName', '32s', cstr),  # (* String32Type *)
        ('AuxFileName', '80s', cstr),  # (* String80Type *)
        ('RootText', '400s', cstr),    # (* String400Type *)
        ('StartTime', 'd'),            # (* LONGREAL *)
        ('MaxSamples', 'i'),           # (* INT32 *)
        ('CRC', 'I'),                  # (* CARD32 *)
        ('Features', 'h'),             # (* SET16 *)
        ('Filler1', 'h', None),        # (* INT16 *)
        ('Filler2', 'i', None),        # (* INT32 *)
        ('RoTcEnumerator', '32h'),     # (* ARRAY[0..Max_TcKind_M1] OF INT16 *)
        ('RoTcKind', '32s', cstr)      # (* ARRAY[0..Max_TcKind_M1] OF INT8 *)
    ]

    required_size = 640

    rectypes = [
        None,
        GroupRecord,
        SeriesRecord,
        SweepRecord,
        TraceRecord
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class StimulationRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),                   # (* INT32 *)
        ('EntryName', '32s', cstr),      # (* String32Type *)
        ('FileName', '32s', cstr),       # (* String32Type *)
        ('AnalName', '32s', cstr),       # (* String32Type *)
        ('DataStartSegment', 'i'),       # (* INT32 *)
        ('DataStartTime', 'd'),          # (* LONGREAL *)
        ('SampleInterval', 'd'),         # (* LONGREAL *)
        ('SweepInterval', 'd'),          # (* LONGREAL *)
        ('LeakDelay', 'd'),              # (* LONGREAL *)
        ('FilterFactor', 'd'),           # (* LONGREAL *)
        ('NumberSweeps', 'i'),           # (* INT32 *)
        ('NumberLeaks', 'i'),            # (* INT32 *)
        ('NumberAverages', 'i'),         # (* INT32 *)
        ('ActualAdcChannels', 'i'),      # (* INT32 *)
        ('ActualDacChannels', 'i'),      # (* INT32 *)
        ('ExtTrigger', 'b'),             # (* BYTE *)
                                         #  ExtTriggerType = ( TrigNone,
                                         #   TrigSeries,
                                         #   TrigSweep,
                                         #   TrigSweepNoLeak );
        ('NoStartWait', '?'),            # (* BOOLEAN *)
        ('UseScanRates', '?'),           # (* BOOLEAN *)
        ('NoContAq', '?'),               # (* BOOLEAN *)
        ('HasLockIn', '?'),              # (* BOOLEAN *)
        ('OldStartMacKind', '?'),        # (* CHAR *)
        ('OldEndMacKind', '?'),          # (* BOOLEAN *)
        ('AutoRange', 'b'),              # (* BYTE *)
                                         #  AutoRangingType = ( AutoRangingOff,
                                         #    AutoRangingPeak,
                                         #    AutoRangingMean,
                                         #    AutoRangingRelSeg );
        ('BreakNext', '?'),              # (* BOOLEAN *)
        ('IsExpanded', '?'),             # (* BOOLEAN *)
        ('LeakCompMode', '?'),           # (* BOOLEAN *)
        ('HasChirp', '?'),               # (* BOOLEAN *)
        ('OldStartMacro', '32s', cstr),  # (* String32Type *)
        ('OldEndMacro', '32s', cstr),    # (* String32Type *)
        ('IsGapFree', '?'),              # (* BOOLEAN *)
        ('HandledExternally', '?'),      # (* BOOLEAN *)
        ('Filler1', '?', None),          # (* BOOLEAN *)
        ('Filler2', '?', None),          # (* BOOLEAN *)
        ('CRC', 'I')                     # (* CARD32 *)
    ]

    required_size = 248


class ChannelRecordStimulus(TreeNode):
    field_info = [
        ('Mark', 'i'),                   # (* INT32 *)
        ('LinkedChannel', 'i'),          # (* INT32 *)
        ('CompressionFactor', 'i'),      # (* INT32 *)
        ('YUnit', '8s', cstr),           # (* String8Type *)
        ('AdcChannel', 'H'),             # (* INT16 *)
        ('AdcMode', 'b', getADCMode),    # (* BYTE *)
                                         # AdcType = ( AdcOff,
                                         #             Analog,
                                         #             Digitals,
                                         #             Digital,
                                         #             AdcVirtual );
        ('DoWrite', '?'),                # (* BOOLEAN *)
        ('LeakStore', 'b'),              # (* BYTE *)
                                         # LeakStoreType = ( LNone,
                                         #                   LStoreAvg,
                                         #                   LStoreEach,
                                         #                   LNoStore );
        ('AmplMode', 'b', getAmplMode),  # (* BYTE *)
                                         # AmplModeType = (AnyAmplMode,
                                         #                 VCAmplMode,
                                         #                 CCAmplMode,
                                         #                 IDensityMode );
        ('OwnSegTime', '?'),             # (* BOOLEAN *)
        ('SetLastSegVmemb', '?'),        # (* BOOLEAN *)
        ('DacChannel', 'H'),             # (* INT16 *)
        ('DacMode', 'b'),                # (* BYTE *)
        ('HasLockInSquare', 'b'),        # (* BYTE *)
        ('RelevantXSegment', 'i'),       # (* INT32 *)
        ('RelevantYSegment', 'i'),       # (* INT32 *)
        ('DacUnit', '8s', cstr),         # (* String8Type *)
        ('Holding', 'd'),                # (* LONGREAL *)
        ('LeakHolding', 'd'),            # (* LONGREAL *)
        ('LeakSize', 'd'),               # (* LONGREAL *)
        ('LeakHoldMode', 'b'),           # (* BYTE *)
                                         # LeakHoldType = ( Labs,
                                         #                  Lrel,
                                         #                  LabsLH,
                                         #                  LrelLH );
        ('LeakAlternate', '?'),          # (* BOOLEAN *)
        ('AltLeakAveraging', '?'),       # (* BOOLEAN *)
        ('LeakPulseOn', '?'),            # (* BOOLEAN *)
        ('StimToDacID', 'H', convertStimToDacID),  # (* SET16 *)
        ('CompressionMode', 'H'),     # (* SET16 *)
                                      # CompressionMode : Specifies how to the data
                                      #    -> meaning of bits:
                                      #       bit 0 (CompReal)   -> high = store as real
                                      #                             low  = store as int16
                                      #       bit 1 (CompMean)   -> high = use mean
                                      #                             low  = use single sample
                                      #       bit 2 (CompFilter) -> high = use digital filter
        ('CompressionSkip', 'i'),     # (* INT32 *)
        ('DacBit', 'H'),              # (* INT16 *)
        ('HasLockInSine', '?'),       # (* BOOLEAN *)
        ('BreakMode', 'b'),           # (* BYTE *)
                                      # BreakType = ( NoBreak,
                                      #               BreakPos,
                                      #               BreakNeg );
        ('ZeroSeg', 'i'),             # (* INT32 *)
        ('StimSweep', 'i'),           # (* INT32 *)
        ('Sine_Cycle', 'd'),          # (* LONGREAL *)
        ('Sine_Amplitude', 'd'),      # (* LONGREAL *)
        ('LockIn_VReversal', 'd'),    # (* LONGREAL *)
        ('Chirp_StartFreq', 'd'),     # (* LONGREAL *)
        ('Chirp_EndFreq', 'd'),       # (* LONGREAL *)
        ('Chirp_MinPoints', 'd'),     # (* LONGREAL *)
        ('Square_NegAmpl', 'd'),      # (* LONGREAL *)
        ('Square_DurFactor', 'd'),    # (* LONGREAL *)
        ('LockIn_Skip', 'i'),         # (* INT32 *)
        ('Photo_MaxCycles', 'i'),     # (* INT32 *)
        ('Photo_SegmentNo', 'i'),     # (* INT32 *)
        ('LockIn_AvgCycles', 'i'),    # (* INT32 *)
        ('Imaging_RoiNo', 'i'),       # (* INT32 *)
        ('Chirp_Skip', 'i'),          # (* INT32 *)
        ('Chirp_Amplitude', 'd'),     # (* LONGREAL *)
        ('Photo_Adapt', 'b'),         # (* BYTE *)
        ('Sine_Kind', 'b'),           # (* BYTE *)
        ('Chirp_PreChirp', 'b'),      # (* BYTE *)
        ('Sine_Source', 'b'),         # (* BYTE *)
        ('Square_NegSource', 'b'),    # (* BYTE *)
        ('Square_PosSource', 'b'),    # (* BYTE *)
        ('Chirp_Kind', 'b', getChirpKind),  # (* BYTE *)
        ('Chirp_Source', 'b'),        # (* BYTE *)
        ('DacOffset', 'd'),           # (* LONGREAL *)
        ('AdcOffset', 'd'),           # (* LONGREAL *)
        ('TraceMathFormat', 'b'),     # (* BYTE *)
        ('HasChirp', '?'),            # (* BOOLEAN *)
        ('Square_Kind', 'b', getSquareKind),  # (* BYTE *)
        ('Filler1', '5s', None),      # (* ARRAY[0..5] OF CHAR *)
        ('Square_BaseIncr', 'd'),     # (* LONGREAL *)
        ('Square_Cycle', 'd'),        # (* LONGREAL *)
        ('Square_PosAmpl', 'd'),      # (* LONGREAL *)
        ('CompressionOffset', 'i'),   # (* INT32 *)
        ('PhotoMode', 'i'),           # (* INT32 *)
        ('BreakLevel', 'd'),          # (* LONGREAL *)
        ('TraceMath', '128s', cstr),  # (* String128Type *)
        ('Filler2', 'i', None),       # (* INT32 *)
        ('CRC', 'I'),                 # (* CARD32 *)
        ('UnknownFiller', '?', None),  # Undocumented
    ]

    required_size = 401


class StimSegmentRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),                               # (* INT32 *)
        ('Class', 'b', getSegmentClass),             # (* BYTE *)
        ('StoreKind', 'b', getStoreType),            # (* BYTE *)
        ('VoltageIncMode', 'b', getIncrementMode),   # (* BYTE *)
        ('DurationIncMode', 'b', getIncrementMode),  # (* BYTE *)
        ('Voltage', 'd'),                            # (* LONGREAL *)
        ('VoltageSource', 'i', getSourceType),       # (* INT32 *)
        ('DeltaVFactor', 'd'),                       # (* LONGREAL *)
        ('DeltaVIncrement', 'd'),                    # (* LONGREAL *)
        ('Duration', 'd'),                           # (* LONGREAL *) [s]
        ('DurationSource', 'i', getSourceType),      # (* INT32 *)
        ('DeltaTFactor', 'd'),                       # (* LONGREAL *)
        ('DeltaTIncrement', 'd'),                    # (* LONGREAL *)
        ('Filler1', 'i', None),                      # (* INT32 *)
        ('CRC', 'I'),                                # (* CARD32 *)
        ('ScanRate', 'd'),                           # (* LONGREAL *)
    ]

    required_size = 80


class StimulusTemplate(TreeNode):
    field_info = [
        ('Version', 'i'),              # (* INT32 *)
        ('Mark', 'i'),                 # (* INT32 *)
        ('VersionName', '32s', cstr),  # (* String32Type *)
        ('MaxSamples', 'i'),           # (* INT32 *)
        ('Filler1', 'i', None),        # (* INT32 *)
        ('StimParams', '10d'),         # (* ARRAY[0..9] OF LONGREAL *), (* StimParams     = 10  *)
        ('StimParamChars', '320s', cstr),  # (* ARRAY[0..9],[0..31]OF CHAR *),(* StimParamChars = 320 *)
        ('Reserved', '128s', None),    # (* String128Type *)
        ('Filler2', 'i', None),        # (* INT32 *)
        ('CRC', 'I')                   # (* CARD32 *)
    ]

    required_size = 584

    rectypes = [
        None,
        StimulationRecord,
        ChannelRecordStimulus,
        StimSegmentRecord,
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


class BundleItem(Struct):
    field_info = [
        ('Start', 'i'),             # (* INT32 *)
        ('Length', 'i'),            # (* INT32 *)
        ('Extension', '8s', cstr),  # (* ARRAY[0..7] OF CHAR *)
    ]

    required_size = 16


class BundleHeader(Struct):
    field_info = [
        ('Signature', '8s', cstr),              # (* ARRAY[0..7] OF CHAR *)
        ('Version', '32s', cstr),               # (* ARRAY[0..31] OF CHAR *)
        ('Time', 'd'),                          # (* LONGREAL *)
        ('Items', 'i'),                         # (* INT32 *)
        ('IsLittleEndian', '?'),                # (* BOOLEAN *)
        ('Reserved', '11s', None),              # (* ARRAY[0..10] OF CHAR *)
        ('BundleItems', BundleItem.array(12)),  # (* ARRAY[0..11] OF BundleItem *)
    ]

    required_size = 256


class AnalysisScalingRecord(Struct):
    field_info = [
        ('MinValue', 'd'),        # (* LONGREAL *)
        ('MaxValue', 'd'),        # (* LONGREAL *)
        ('GridFactor', 'd'),      # (* LONGREAL *)
        ('TicLength', 'h'),       # (* INT16 *)
        ('TicNumber', 'h'),       # (* INT16 *)
        ('TicDirection', 'b'),    # (* BYTE *)
        ('AxisLevel', 'b'),       # (* BYTE *)
        # = ( Min, Zero, Max );
        ('AxisType', 'b'),        # (* BYTE *)
        # = ( ScaleLinear,
        #     ScaleLog,
        #     ScaleInverse,
        #     ScaleSqrt,
        #     ScaleSquare );
        ('ScaleMode', 'b'),       # (* BYTE *)
        # = ( ScaleFixed, ScaleSeries, ScaleSweeps );
        ('NoUnit', '?'),          # (* BOOLEAN *)
        ('Obsolete', '?', None),  # (* BOOLEAN *)
        ('ZeroLine', '?'),        # (* BOOLEAN *)
        ('Grid', '?'),            # (* BOOLEAN *)
        ('Nice', '?'),            # (* BOOLEAN *)
        ('Label', '?'),           # (* BOOLEAN *)
        ('Centered', '?'),        # (* BOOLEAN *)
        ('IncludeZero', '?')      # (* BOOLEAN *)
    ]

    required_size = 40


class AnalysisEntryRecord(Struct):
    field_info = [
        ('XWave', 'h'),             # (* INT16 *)
        ('YWave', 'h'),             # (* INT16 *)
        ('MarkerSize', 'h'),        # (* INT16 *)
        ('MarkerColorRed', 'H'),    # (* CARD16 *)
        ('MarkerColorGreen', 'H'),  # (* CARD16 *)
        ('MarkerColorBlue', 'H'),   # (* CARD16 *)
        ('MarkerKind', 'b'),        # (* BYTE *)
        # = ( MarkerPoint,
        #     MarkerPlus,
        #     MarkerStar,
        #     MarkerDiamond,
        #     MarkerX,
        #     MarkerSquare );
        ('EActive', '?'),           # (* BOOLEAN *)
        ('Line', '?'),              # (* BOOLEAN *)
        ('TraceColor', '?')         # (* BOOLEAN *)
    ]

    required_size = 16


class AnalysisGraphRecord(Struct):
    field_info = [
        ('GActive', '?'),                     # (* BOOLEAN *)
        ('Overlay', '?'),                     # (* BOOLEAN *)
        ('Wrap', 'b'),                        # (* CHAR *)
        ('OvrlSwp', '?'),                     # (* BOOLEAN *)
        ('Normalize', 'b'),                   # (* BYTE *)
        # = ( NormalizeNone, NormalizeMax, NormalizeMinMax );
        ('Spare1', 'b', None),                # (* BYTE *)
        ('Spare2', 'b', None),                # (* BYTE *)
        ('Spare3', 'b', None),                # (* BYTE *)
        ('XScaling', AnalysisScalingRecord),  # (* ScalingRecord *)
        ('YScaling', AnalysisScalingRecord),  # (* ScalingRecord *)
        ('Entry0', AnalysisEntryRecord),      # (* EntryRecSize *)
        ('Entry1', AnalysisEntryRecord),      # (* EntryRecSize *)
        ('Entry2', AnalysisEntryRecord),      # (* EntryRecSize *)
        ('Entry3', AnalysisEntryRecord)       # (* EntryRecSize *)
    ]

    required_size = 152


class AnalysisFunctionRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),              # (* INT32 *)
        ('Name', '32s', cstr),      # (* String32Size *)
        ('Unit', '8s', cstr),       # (* String8Size *)
        ('LeftOperand', 'h'),       # (* INT16 *)
        ('RightOperand', 'h'),      # (* INT16 *)
        ('LeftBound', 'd'),         # (* LONGREAL *)
        ('RightBound', 'd'),        # (* LONGREAL *)
        ('Constant', 'd'),          # (* LONGREAL *)
        ('XSegmentOffset', 'i'),    # (* INT32 *)
        ('YSegmentOffset', 'i'),    # (* INT32 *)
        ('TcEnumarator', 'h'),      # (* INT16  *)
        ('Function', 'b'),          # (* BYTE *)
        # = ( SweepCountAbsc,     (* general *)
        #     TimeAbsc,
        #     TimerAbsc,
        #     RealtimeAbsc,
        #     SegAmplitude,       (* X segment property *)
        #     SegDuration,
        #     ScanRateAbsc,
        #     ExtremumMode,       (* Y analysis *)
        #     MaximumMode,
        #     MinimumMode,
        #     MeanMode,
        #     IntegralMode,
        #     VarianceMode,
        #     SlopeMode,
        #     TimeToExtrMode,
        #     AnodicChargeMode,   (* potentiostat *)
        #     CathodChargeMode,
        #     CSlowMode,       (* potmaster: spare *)
        #     RSeriesMode,     (* potmaster: spare *)
        #     UserParam1Mode,
        #     UserParam2Mode,
        #     LockInCMMode,       (* lock-in *)
        #     LockInGMMode,
        #     LockInGSMode,
        #     SeriesTime,         (* misk *)
        #     StimulusMode,
        #     SegmentTimeAbs,
        #     OpEquationMode,     (* math *)
        #     ConstantMode,
        #     OperatorPlusMode,
        #     OperatorMinusMode,
        #     OperatorMultMode,
        #     OperatorDivMode,
        #     OperatorAbsMode,
        #     OperatorLogMode,
        #     OperatorSqrMode,
        #     OperatorInvMode,
        #     OperatorInvLogMode,
        #     OperatorInvSqrMode,
        #     TraceMode,          (* trace *)
        #     QMode,
        #     InvTraceMode,
        #     InvQMode,
        #     LnTraceMode,
        #     LnQMode,
        #     LogTraceMode,
        #     LogQMode,
        #     TraceXaxisMode,
        #     FreqMode,           (* spectra *)
        #     DensityMode,
        #     HistoAmplMode,      (* histogram *)
        #     HistoBinsMode,
        #     OnlineIndex,
        #     ExtrAmplMode,
        #     SegmentTimeRel,
        #     CellPotential,   (* potmaster: OCP *)
        #     SealResistance,  (* potmaster: ElectrodeArea *)
        #     RsValue,         (* potmaster: spare *)
        #     GLeak,           (* potmaster: spare *)
        #     MConductance,    (* potmaster: spare *)
        #     Temperature,
        #     PipettePressure, (* potmaster: spare *)
        #     InternalSolution,
        #     ExternalSolution,
        #     DigitalIn,
        #     OperatorBitInMode,
        #     ReversalMode,
        #     LockInPhase,
        #     LockInFreq,
        #     TotMeanMode,     (* obsolete: replaced by MeanMode + CursorKind *)
        #     DiffMode,
        #     IntSolValue,
        #     ExtSolValue,
        #     OperatorAtanMode,
        #     OperatorInvAtanMode,
        #     TimeToMinMode,
        #     TimeToMaxMode,
        #     TimeToThreshold,
        #     TraceEquationMode,
        #     ThresholdAmpl,
        #     XposMode,
        #     YposMode,
        #     ZposMode,
        #     TraceCountMode,
        #     AP_Baseline,
        #     AP_MaximumAmpl,
        #     AP_MaximumTime,
        #     AP_MinimumAmpl,
        #     AP_MinimumTime,
        #     AP_RiseTime1Dur,
        #     AP_RiseTime1Slope,
        #     AP_RiseTime1Time,
        #     AP_RiseTime2Dur,
        #     AP_RiseTime2Slope,
        #     AP_RiseTime2Time,
        #     AP_Tau,
        #     MatrixXindexMode,
        #     MatrixYindexMode,
        #     YatX_Mode,
        #     ThresholdCount,
        #     SECM_3Dx,
        #     SECM_3Dy,
        #     InterceptMode,
        #     MinAmplMode,
        #     MaxAmplMode,
        #     TauMode );
        ('DoNotebook', '?'),        # (* BOOLEAN *)
        ('NoFit', '?'),             # (* BOOLEAN *)
        ('NewName', '?'),           # (* BOOLEAN *)
        ('TargetValue', 'h'),       # (* INT16 *)
        ('CursorKind', 'b'),        # (* BYTE *)
        # = ( Cursor_Segment,     (* cursor relative to segment *)
        #     Cursor_Trace );     (* cursor relative to trace *)
        ('TcKind1', 'b'),           # (* BYTE *)
        # = ( TicLeft, TicRight, TicBoth );
        ('TcKind2', 'b'),           # (* BYTE *)
        # = ( TicLeft, TicRight, TicBoth );
        ('CursorSource', 'b'),      # (* BYTE *)
        ('CRC', 'I'),               # (* CARD32 *)
        ('Equation', '64s', cstr),  # (* String64Size *)
        ('BaselineMode', 'b'),      # (* BYTE *)
        #  = ( Baseline_Zero,      (* baseline relative to zero *)
        #      Baseline_Cursors,   (* baseline = intersection with cursors *)
        #      Baseline_Auto );    (* baseline = intersection with cursors *)
        ('SearchDirection', 'b'),   # (* BYTE *)
        #  = ( Search_All,
        #      Search_Positive,
        #      Search_Negative );
        ('SourceValue', 'h'),       # (* INT16 *)
        ('CursorAnker', 'h'),       # (* INT16 *)
        ('Spare1', 'h', None),      # (* INT16 *)
    ]

    required_size = 168


class AnalysisMethodRecord(TreeNode):
    field_info = [
        ('Mark', 'i'),                              # (* INT32 *)
        ('EntryName', '32s', cstr),                 # (* String32Size *)
        ('SharedXWin1', '?'),                       # (* BOOLEAN *)
        ('SharedXWin2', '?'),                       # (* BOOLEAN *)
        ('m1', '?'),                                # (* BOOLEAN *)
        ('m2', '?'),                                # (* BOOLEAN *)
        ('Graph0', AnalysisGraphRecord.array(12)),  # (* MaxGraphs * GraphRecSize', ' '),1824 *)
        ('m3', 'i'),                                # (* INT32 *)
        ('CRC', 'I'),                               # (* CARD32 *)
        ('Headers', '384s', cstr),                  # (* MaxGraphs * String32Size', ' '), 384 *)
        ('LastXmin', '12d'),                        # (* MaxGraphs * LONGREAL', ' '),  96 *)
        ('LastXmax', '12d'),                        # (* MaxGraphs * LONGREAL', ' '),  96 *)
        ('LastYmin', '12d'),                        # (* MaxGraphs * LONGREAL', ' '),  96 *)
        ('LastYmax', '12d'),                        # (* MaxGraphs * LONGREAL', ' '),  96 *)
    ]

    required_size = 2640


class Analysis(TreeNode):
    field_info = [
        ('Version', 'i'),              # (* INT32 *)
        ('Mark', 'i'),                 # (* INT32 *)
        ('VersionName', '32s', cstr),  # (* String32Size *)
        ('Obsolete', 'b', None),       # (* BYTE *)
        ('MaxTraces', 'c'),            # (* CHAR *)
        ('WinDefined', '?'),           # (* BOOLEAN *)
        ('rt1', 'b'),                  # (* BYTE *)
        ('CRC', 'I'),                  # (* CARD32 *)
        ('WinNr', '12b'),              # (* MaxFileGraphs *)
        ('rt2', 'i')                   # (* INT32 *)
    ]

    required_size = 64

    rectypes = [
        None,
        AnalysisMethodRecord,
        AnalysisFunctionRecord
    ]

    def __init__(self, fh, endianess):
        TreeNode.__init__(self, fh, endianess, self.rectypes, None)


# Not a TreeNode as the data starts at file offset `trace.Data`
# and also endianess handling is different
class RawData():
    def __init__(self, bundle):
        self.bundle = bundle

    def __getitem__(self, *args):
        """
        Get a specific data block as numpy array.

        :param args: Can be either a `TraceRecord` or a list holding four indizes
                     (group, series, sweep, trace).

        :return: 1D-numpy array
        """

        if isinstance(args[0], TraceRecord):
            trace = args[0]
        else:
            index = list(*args)
            assert len(index) == 4, f"Unexpected list format with {len(index)} items."
            pul = self.bundle.pul
            trace = pul[index[0]][index[1]][index[2]][index[3]]

        assert trace.DataKind["IsLittleEndian"], "Big endian support is not implemented"

        with self.bundle:
            self.bundle.fh.seek(trace.Data)
            dtype = convertDataFormatToNP(trace.DataFormat)
            data = np.fromfile(self.bundle.fh, count=trace.DataPoints, dtype=dtype)
            return data * trace.DataScaler + trace.ZeroData

    def __str__(self):
        return "RawData(...)"
