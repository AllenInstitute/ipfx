import math

from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
from pynwb.form.backends.hdf5.h5_utils import H5DataIO

PLACEHOLDER = "PLACEHOLDER"
V_CLAMP_MODE = 0
I_CLAMP_MODE = 1


# TODO Use the pint package if doing that manually gets too involved
def parseUnit(unitString):
    """
    Split a SI unit string with prefix into the base unit and the prefix (as number).
    """

    if unitString == "pA":
        return 1e-12, "A"
    elif unitString == "A":
        return 1.0, "A"
    elif unitString == "mV":
        return 1e-3, "V"
    elif unitString == "V":
        return 1.0, "V"
    else:
        raise ValueError(f"Unsupported unit string {unitString}.")


def getStimulusSeriesClass(clampMode):
    """
    Return the appropriate pynwb stimulus class for the given clamp mode.
    """

    if clampMode == V_CLAMP_MODE:
        return VoltageClampStimulusSeries
    elif clampMode == I_CLAMP_MODE:
        return CurrentClampStimulusSeries
    else:
        raise ValueError(f"Unsupported clamp mode {clampMode}.")


def getAcquiredSeriesClass(clampMode):
    """
    Return the appropriate pynwb acquisition class for the given clamp mode.
    """

    if clampMode == V_CLAMP_MODE:
        return VoltageClampSeries
    elif clampMode == I_CLAMP_MODE:
        return CurrentClampSeries
    else:
        raise ValueError(f"Unsupported clamp mode {clampMode}.")


def createSeriesName(prefix, number, total):
    """
    Format a unique series group name of the form `prefix_XXX` where `XXX` is
    the formatted `number` long enough for `total` number of groups.
    """

    return f"{prefix}_{number:0{math.ceil(math.log(total, 10))}d}", number + 1


def createCompressedDataset(array):
    """
    Request compression for the given array and return it wrapped.
    """

    return H5DataIO(data=array, compression=True, chunks=True, shuffle=True, fletcher32=True)
