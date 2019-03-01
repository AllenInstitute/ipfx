import warnings

from ipfx.x_to_nwb.hr_nodes import (Pulsed, StimulusTemplate, AmplifierFile,
                                    ProtocolMethod, Solutions, Marker,
                                    BundleHeader, Analysis, RawData)


class Bundle():
    """
    Represent a PATCHMASTER tree file in memory
    """

    item_classes = {
        '.pul': Pulsed,
        '.dat': RawData,
        '.pgf': StimulusTemplate,
        '.amp': AmplifierFile,
        '.mth': ProtocolMethod,
        '.sol': Solutions,
        '.mrk': Marker,
        '.onl': Analysis
    }

    def __init__(self, file_name):
        self.file_name = file_name

        with self:
            if self.fh.read(4) != b'DAT2':
                raise ValueError(f"No support for other files than 'DAT2' format")

            self.fh.seek(0)

            # Read header assuming little endian
            endian = '<'
            self.header = BundleHeader(self.fh, endian)

            # If the header is bad, re-read using big endian
            if not self.header.IsLittleEndian:
                endian = '>'
                self.fh.seek(0)
                self.header = BundleHeader(self.fh, endian)

            # catalog extensions of bundled items
            self.catalog = {}
            for item in self.header.BundleItems:
                item.instance = None
                ext = item.Extension
                self.catalog[ext] = item

            if not self.header.Version.startswith("v2x90"):
                warnings.warn(f"The DAT file version '{self.header.Version}' of '{file_name}' might "
                              "be incompatible and therefore read/interpretation errors are possible.")

        return

    def __enter__(self):
        self.fh = open(self.file_name, 'rb')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fh.close()

    @property
    def data(self):
        """
        The Data object from this bundle.
        """
        return self._get_item_instance('.dat')

    @property
    def pul(self):
        """
        The Pulsed object from this bundle.
        """
        return self._get_item_instance('.pul')

    @property
    def pgf(self):
        """
        The Stimulus Template object from this bundle.
        """
        return self._get_item_instance('.pgf')

    @property
    def sol(self):
        """
        The Solutions object from this bundle.
        """
        return self._get_item_instance('.sol')

    @property
    def onl(self):
        """
        The Online Analysis object from this bundle.
        """
        return self._get_item_instance('.onl')

    @property
    def mth(self):
        """
        The ProtocolMethod object from this bundle.
        """
        return self._get_item_instance('.mth')

    @property
    def mrk(self):
        """
        The Markers object from this bundle.
        """
        return self._get_item_instance('.mrk')

    @property
    def amp(self):
        """
        The Amplifier object from this bundle.
        """
        return self._get_item_instance('.amp')

    # "ana" which holds results from Fitmaster is not supported

    def _get_item_instance(self, ext):
        item = self.catalog.get(ext)

        if item is None:
            return None
        elif item.Length == 0:
            return None
        elif item.instance is not None:
            return item.instance

        cls = self.item_classes[ext]

        with self:
            if ext == '.dat':
                item.instance = cls(self)
            else:
                self.fh.seek(item.Start)
                # read endianess magic
                magic = self.fh.read(4)
                if magic == b'eerT':
                    endianess = '<'
                elif magic == b'Tree':
                    endianess = '>'
                else:
                    raise RuntimeError('Bad file magic: %s' % magic)

                item.instance = cls(self.fh, endianess)

            return item.instance

    def __str__(self):
        return "Bundle(%s)" % list(self.catalog.keys())

    def _all_info(self, outputFile=None):
        """
        Development helper routine, outputs all metadata of the DAT file (bundle)
        either to stdout or the given `outputFile`.
        """

        if outputFile is not None:
            fh = open(outputFile, 'w')
        else:
            fh = None

        print(self.header, file=fh)
        print(self.data, file=fh)
        print(self.pul, file=fh)
        print(self.pgf, file=fh)
        print(self.amp, file=fh)
        print(self.mth, file=fh)
        print(self.sol, file=fh)
        print(self.mrk, file=fh)
        print(self.onl, file=fh)

        print(self.data[0, 0, 0, 0], file=fh)

        print('#' * 80, file=fh)

        # Root
        #   Groups
        #     Series (contains AmplifierState)
        #       Sweeps
        #         Traces

        for group in self.pul:
            print(group, file=fh)
            for series in group:
                print(series, file=fh)
                for sweep in series:
                    print(sweep, file=fh)
                    for trace in sweep:
                        print(trace, file=fh)

        print('#' * 80, file=fh)

        # Root
        #   Stimulation
        #     Channel
        #       StimSegment

        for stimulation in self.pgf:
            print(stimulation, file=fh)
            for channel in stimulation:
                print(channel, file=fh)
                for stimsegment in channel:
                    print(stimsegment, file=fh)

        # Analysis
        #   Method
        #       Function

        if self.onl:
            for method in self.onl:
                print(method, file=fh)
                for function in method:
                    print(function, file=fh)

        # Amplifier
        #   Record
        #       State

        if self.amp:
            for record in self.amp:
                print(record, file=fh)
                for state in record:
                    print(state, file=fh)
