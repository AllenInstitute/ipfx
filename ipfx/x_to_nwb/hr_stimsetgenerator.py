import numpy as np

from ipfx.x_to_nwb.hr_segments import getSegmentClass


class StimSetGenerator:
    """
    High level class for creating stimsets
    """

    def __init__(self, bundle):

        self.bundle = bundle
        self.cache = {}
        pass

    def fetch(self, idx):
        """
        Fetch a stimulus set from the cache, generate it if it is not present.

        Parameter: idx
                This is the stimulus index in the DAT file (SweepRecord.StimCount, converted to 0-base)

        Return: python list of numpy arrays, one array per sweep
        """

        entry = self.cache.get(idx)

        if entry:
            return entry
        elif entry is False:
            return []

        # PGF hierarchy
        #
        # Root
        #  stimRec
        #   channelRec
        #    segmentRec

        try:
            stimRec = self.bundle.pgf[idx]

            if len(stimRec) > 1:
                raise ValueError(f"Unsupported stimRec length {len(stimRec)} for index {idx}.")
            elif stimRec.ActualAdcChannels != 1 or stimRec.ActualDacChannels != 1:
                raise ValueError(f"Unsupported ActualAdcChannel/ActualDacChannels lengths for index {idx}.")

            channelRec = stimRec[0]
            allSweeps = []

            for sweep in range(stimRec.NumberSweeps):
                stimset = np.empty([0])
                for segmentRec in channelRec:
                    cls = getSegmentClass(stimRec, channelRec, segmentRec)
                    # print(cls)
                    segment = cls.createArray(sweep)

                    stimset = np.append(stimset, segment, axis=0)

                allSweeps.append(stimset)

            self.cache[idx] = allSweeps

        except (ValueError, IndexError) as e:
            print(e)
            self.cache[idx] = False
            return []

        return self.cache[idx]
