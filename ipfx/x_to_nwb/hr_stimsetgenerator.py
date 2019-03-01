import numpy as np

from ipfx.x_to_nwb.hr_segments import getSegmentClass
from ipfx.x_to_nwb.conversion_utils import getChannelRecordIndex, getStimulusRecordIndex


class StimSetGenerator:
    """
    High level class for creating stimsets
    """

    def __init__(self, bundle):

        self.bundle = bundle
        self.cache = {}
        pass

    def fetch(self, sweep, trace):
        """
        Fetch a stimulus set from the cache, generate it if it is not present.

        :param sweep: SweepRecord node
        :param trace: TraceRecord node

        Return: python list of numpy arrays, one array per sweep
        """

        # PGF hierarchy
        #
        # Root
        #  stimRec
        #   channelRec
        #    segmentRec

        try:

            stimRec_idx = getStimulusRecordIndex(sweep)
            key = None
            channelRec_index = getChannelRecordIndex(self.bundle.pgf, sweep, trace)

            if channelRec_index is None:
                raise ValueError(f"Could not find a ChannelRecord for the given trace.")

            key = f"{stimRec_idx}.{channelRec_index}"
            entry = self.cache.get(key)

            if entry:
                return entry
            elif entry is False:
                return []

            stimRec = self.bundle.pgf[stimRec_idx]
            channelRec = stimRec[channelRec_index]

            if stimRec.ActualDacChannels != 1:
                raise ValueError(f"Unsupported ActualDacChannels lengths for "
                                 f"sweep index {stimRec_idx} and ChannelRecord index {channelRec_index}.")

            allSweeps = []

            for sweep in range(stimRec.NumberSweeps):
                stimset = np.empty([0])
                for segmentRec in channelRec:
                    cls = getSegmentClass(stimRec, channelRec, segmentRec)
                    # print(cls)
                    segment = cls.createArray(sweep)

                    stimset = np.append(stimset, segment, axis=0)

                allSweeps.append(stimset)

            self.cache[key] = allSweeps

        except (ValueError, IndexError) as e:
            print(e)
            if key:
                self.cache[key] = False
            return []

        return self.cache[key]
