import math
from pkg_resources import get_distribution, DistributionNotFound
import os
from subprocess import Popen, PIPE

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


def createCycleID(numbers, total):
    """
    Create an integer from all numbers which is unique for that combination.

    :param: numbers:
        Iterable holding non-negative integer numbers

    :param: total:
        Total number of TimeSeries written to the NWB file
    """

    assert total > 0, f"Unexpected value for total {total}"

    places = max(math.ceil(math.log(total, 10)), 1)

    result = 0

    for idx, n in enumerate(reversed(numbers)):
        assert n >= 0, f"Unexpected value {n} at index {idx}"
        assert n < 10**places, f"Unexpected value {n} which is larger than {total}"

        result += n * (10**(idx * places))

    return result


def createCompressedDataset(array):
    """
    Request compression for the given array and return it wrapped.
    """

    return H5DataIO(data=array, compression=True, chunks=True, shuffle=True, fletcher32=True)


def getPackageInfo():
    """
    Return a dictionary with version information for the allensdk package
    """

    def get_git_version():
        """
        Returns the project version as derived by git.
        """

        path = os.path.dirname(__file__)
        branch = Popen(f'git -C "{path}" rev-parse --abbrev-ref HEAD', stdout=PIPE,
                       shell=True).stdout.read().rstrip().decode('ascii')
        rev = Popen(f'git -C "{path}" describe --always --tags', stdout=PIPE,
                    shell=True).stdout.read().rstrip().decode('ascii')

        if branch.startswith('fatal') or rev.startswith('fatal'):
            raise ValueError("Could not determine git version")

        return f"({branch}) {rev}"

    try:
        package_version = get_distribution('allensdk').version
    except DistributionNotFound:  # not installed as a package
        package_version = None

    try:
        git_version = get_git_version()
    except ValueError:  # not in a git repostitory
        git_version = None

    version_info = {"repo": "https://github.com/AllenInstitute/ipfx",
                    "package_version": "Unknown",
                    "git_revision": "Unknown"}

    if package_version:
        version_info["package_version"] = package_version

    if git_version:
        version_info["git_revision"] = git_version

    return version_info


def getStimulusRecordIndex(sweep):
    return sweep.StimCount - 1


def getChannelRecordIndex(pgf, sweep, trace):
    """
    Given a pgf node, a SweepRecord and TraceRecord this returns the
    corresponding `ChannelRecordStimulus` node as index.
    """

    stimRec = pgf[getStimulusRecordIndex(sweep)]

    for idx, channelRec in enumerate(stimRec):
        if channelRec.AdcChannel == trace.AdcChannel:
            return idx

    return None
