import ipfx.epochs as ep


class Sweep(object):

    # nwbEpochs are selected/looked up under their ShortName tag prefixed
    # with this string (e.g. "E0" -> "nwb:E0"). This namespaces them away
    # from the legacy epoch names ("test"/"sweep"/"recording"/"stim"/
    # "experiment"), making a name collision between the two structurally
    # impossible.
    NWB_EPOCH_PREFIX = "nwb:"

    def __init__(
        self, t, v, i, clamp_mode, sampling_rate, sweep_number=None,
        epochs=None, nwb_epochs=None
    ):
        self._t = t
        self._v = v
        self._i = i
        self.sampling_rate = sampling_rate
        self.sweep_number = sweep_number
        self.clamp_mode = clamp_mode
        if epochs:
            self.epochs = epochs
        else:
            self.epochs = {}

        self.nwb_epochs = list(nwb_epochs) if nwb_epochs else []
        self._nwb_epoch_lookup = self._build_nwb_epoch_lookup(self.nwb_epochs)

        if self.clamp_mode == "CurrentClamp":
            self._response = self._v
            self._stimulus = self._i
        else:
            self._response = self._i
            self._stimulus = self._v

        self.detect_epochs()
        self._check_epoch_name_collisions()
        self.selected_epoch_name = "recording"

    @staticmethod
    def _build_nwb_epoch_lookup(nwb_epochs):
        """
        Index nwbEpochs by their pre-computed, already-`NWB_EPOCH_PREFIX`-ed
        `name` field (e.g. "nwb:E0"), so they can be selected by name the
        same way the legacy epochs dict is, without risking a name collision
        with a legacy epoch name. The prefix itself is applied once, at load
        time, by whoever produces these records (e.g. MIESNWBData), rather
        than recomputed here on every Sweep construction. A record with no
        `name` (i.e. no `ShortName` tag at load time) is kept in
        self.nwb_epochs but is not selectable by name.

        Raises
        ------
        ValueError
            If two nwbEpoch records in this sweep share the same name --
            selecting by that name would be ambiguous.
        """
        lookup = {}
        for record in nwb_epochs:
            key = record.get("name")
            if not key:
                continue
            if key in lookup:
                raise ValueError(
                    f"Duplicate nwbEpoch name {key!r} within a single sweep "
                    "-- cannot unambiguously select it by name."
                )
            lookup[key] = record
        return lookup

    def _check_epoch_name_collisions(self):
        """
        Guard against a nwbEpoch's (prefixed) lookup key colliding with one
        of the legacy (algorithmically-detected) epoch names. In practice
        this can no longer happen, since nwbEpoch keys always carry the
        `NWB_EPOCH_PREFIX` namespace while legacy epoch names never do --
        this check is kept as a defensive safety net.

        Raises
        ------
        ValueError
            If any name is present in both the legacy epochs and nwbEpochs.
        """
        collisions = set(self.epochs) & set(self._nwb_epoch_lookup)
        if collisions:
            raise ValueError(
                f"Epoch name(s) {sorted(collisions)} exist in both the "
                "legacy epochs and nwbEpochs for this sweep -- rename the "
                "colliding nwbEpoch(s) or resolve the ambiguity before "
                "selecting by name."
            )

    def get_epoch_range(self, name):
        """
        Resolve an epoch name (legacy or nwbEpoch) to a (start_idx, end_idx)
        index range, checking the legacy epochs first and then nwbEpochs.

        Parameters
        ----------
        name: str
            A legacy epoch name (e.g. "recording", "experiment") or a
            nwbEpoch's prefixed ShortName (e.g. "nwb:E0", "nwb:TP_B0").

        Returns
        -------
        (start_idx, end_idx): int tuple

        Raises
        ------
        KeyError
            If `name` is not a known legacy epoch or nwbEpoch ShortName.
        """
        if name in self.epochs:
            return self.epochs[name]
        if name in self._nwb_epoch_lookup:
            record = self._nwb_epoch_lookup[name]
            return record["start_idx"], record["end_idx"]
        raise KeyError(
            f"{name!r} is not a known epoch name (checked legacy epochs "
            f"{sorted(self.epochs)} and nwbEpochs "
            f"{sorted(self._nwb_epoch_lookup)})"
        )

    def get_nwb_epoch(self, name):
        """
        Return the raw nwbEpoch record (start_time/stop_time/start_idx/
        end_idx/treelevel/tags) for `name` (its ShortName tag, prefixed with
        `NWB_EPOCH_PREFIX`, e.g. "nwb:E0").

        Raises
        ------
        KeyError
            If `name` is not a known nwbEpoch.
        """
        return self._nwb_epoch_lookup[name]

    @property
    def t(self):
        start_idx, end_idx = self.get_epoch_range(self.selected_epoch_name)
        return self._t[start_idx:end_idx+1]

    @property
    def v(self):
        start_idx, end_idx = self.get_epoch_range(self.selected_epoch_name)
        return self._v[start_idx:end_idx+1]

    @property
    def i(self):
        start_idx, end_idx = self.get_epoch_range(self.selected_epoch_name)
        return self._i[start_idx:end_idx+1]

    def select_epoch(self, epoch_name):
        # validate eagerly, legacy or nwbEpoch
        self.get_epoch_range(epoch_name)
        self.selected_epoch_name = epoch_name

    def set_time_zero_to_index(self, time_step):
        dt = 1. / self.sampling_rate
        self._t = self._t - time_step*dt

    def detect_epochs(self):
        """
        Detect epochs if they are not provided in the constructor

        """

        if "test" not in self.epochs:
            self.epochs["test"] = ep.get_test_epoch(self._stimulus, self.sampling_rate)
        if self.epochs["test"]:
            test_pulse = True
        else:
            test_pulse = False

        if "sweep" not in self.epochs:
            self.epochs["sweep"] = ep.get_sweep_epoch(self._i)
        if "recording" not in self.epochs:
            self.epochs["recording"] = ep.get_recording_epoch(self._response)
        # get valid recording by selecting epoch and using i/v prop before detecting stim
        self.select_epoch("recording")
        stim = self.i if self.clamp_mode == "CurrentClamp" else self.v
        if "stim" not in self.epochs:
            self.epochs["stim"] = ep.get_stim_epoch(stim, test_pulse)
        if "experiment" not in self.epochs:
            self.epochs["experiment"] = ep.get_experiment_epoch(stim, self.sampling_rate, test_pulse)


class SweepSet(object):
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def _prop(self, prop):
        return [getattr(s, prop) for s in self.sweeps]

    def select_epoch(self, epoch_name):
        for sweep in self.sweeps:
            sweep.select_epoch(epoch_name)

    def align_to_start_of_epoch(self, epoch_name):

        for sweep in self.sweeps:
            start_idx, end_idx = sweep.get_epoch_range(epoch_name)
            sweep.set_time_zero_to_index(start_idx)

    @property
    def t(self):
        return self._prop('t')

    @property
    def v(self):
        return self._prop('v')

    @property
    def i(self):
        return self._prop('i')

    @property
    def sweep_number(self):
        return self._prop('sweep_number')

    @property
    def sampling_rate(self):
        return self._prop('sampling_rate')
