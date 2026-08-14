from typing import Dict, Any, List

from ipfx.stimulus import StimulusOntology
from ipfx.dataset.labnotebook import LabNotebookReader
from ipfx.dataset.ephys_nwb_data import EphysNWBData, get_finite_or_none
from ipfx.sweep import Sweep


class MIESNWBData(EphysNWBData):
    """
    Provides an Ephys Data Interface to a MIES generated NWB file

    """

    def __init__(
            self,
            nwb_file: str,
            notebook: LabNotebookReader,
            ontology: StimulusOntology,
            load_into_memory: bool = True,
            validate_stim: bool = True
    ):
        super(MIESNWBData, self).__init__(
            nwb_file=nwb_file,
            ontology=ontology,
            load_into_memory=load_into_memory,
            validate_stim=validate_stim
        )
        self.notebook = notebook

    def get_stim_code_ext(self, sweep_number):
        stim_code = super().get_stimulus_code(sweep_number)

        cnt = self.notebook.get_value("Set Sweep Count", sweep_number, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)
        return stim_code_ext

    def get_sweep_metadata(self, sweep_number: int) -> Dict[str, Any]:
        attrs = self.get_sweep_attrs(sweep_number)

        sweep_record = {
            "sweep_number": sweep_number,
            "stimulus_units": self.get_stimulus_unit(sweep_number),
            "bridge_balance_mohm": get_finite_or_none(attrs, "bridge_balance"),
            "leak_pa": get_finite_or_none(attrs, "bias_current"),
            "stimulus_scale_factor": self.notebook.get_value(
                "Scale Factor", sweep_number, None
            ),
            "stimulus_code": self.get_stimulus_code(sweep_number),
            "stimulus_code_ext": self.get_stim_code_ext(sweep_number),
            "clamp_mode": self.get_clamp_mode(sweep_number)
        }

        if self.ontology:
            sweep_record["stimulus_name"] = self.get_stimulus_name(
                sweep_record["stimulus_code"]
            )

        return sweep_record

    def get_nwb_epochs(self, sweep_number: int) -> List[Dict[str, Any]]:
        """
        Extract the NWB TimeIntervals ("epochs") rows associated with a given
        sweep's stimulus series.

        This is a distinct concept from EphysDataInterface/Sweep's own
        "epochs" (the algorithmically-detected QC alignment windows -- test/
        sweep/recording/stim/experiment -- computed straight from the trace
        data). This instead reads the MIES-authored, tag-annotated intervals
        table written into the NWB file's own `epochs` (nwbfile.epochs), and
        is referred to as "nwbEpochs" throughout to avoid confusing the two.
        Only MIES-generated NWB files are expected to have this table
        structured this way -- this method therefore lives on MIESNWBData
        only, not the shared EphysNWBData base or HBGNWBData.

        Parameters
        ----------
        sweep_number

        Returns
        -------
        A list of records, one per nwbfile.epochs row whose `timeseries`
        column references this sweep's stimulus series, each of the form:
            {
                "start_time": float,   # seconds, absolute session time
                "stop_time": float,    # seconds, absolute session time
                "start_idx": int,      # sample index into this sweep's trace
                "end_idx": int,        # sample index into this sweep's trace
                "treelevel": int,      # MIES epoch-hierarchy nesting depth
                "tags": Dict[str, str],  # parsed "Key=Value" tags
                "name": Optional[str],  # tags["ShortName"], already prefixed
                                         # with Sweep.NWB_EPOCH_PREFIX for
                                         # by-name lookup on a Sweep, or None
                                         # if there is no ShortName tag.
            }
        The "name" prefixing is done once here, at load time, rather than
        every time a Sweep is constructed from this same data.

        Empty list if the file has no epochs table, or none of its rows
        reference this sweep's stimulus series.
        """
        # EphysNWBData.nwb is assigned outside __init__ (in load_nwb), which
        # mypy can't type-infer -- the same pre-existing gap already affects
        # other self.nwb accesses in ephys_nwb_data.py (e.g. _get_series,
        # get_sweep_numbers).
        table = self.nwb.epochs  # type: ignore[has-type]
        if table is None:
            return []

        stimulus_series = self._get_series(sweep_number, self.STIMULUS)

        records = []
        for row_idx in range(len(table)):
            for ref in table["timeseries"][row_idx]:
                if ref.timeseries is not stimulus_series:
                    continue

                tags = self._parse_tags(table["tags"][row_idx])
                short_name = tags.get("ShortName")

                records.append({
                    "start_time": float(table["start_time"][row_idx]),
                    "stop_time": float(table["stop_time"][row_idx]),
                    "start_idx": int(ref.idx_start),
                    "end_idx": int(ref.idx_start) + int(ref.count),
                    "treelevel": int(table["treelevel"][row_idx]),
                    "tags": tags,
                    "name": (
                        f"{Sweep.NWB_EPOCH_PREFIX}{short_name}"
                        if short_name else None
                    ),
                })

        return records

    @staticmethod
    def _parse_tags(tags) -> Dict[str, str]:
        """
        Parse a nwbEpoch row's raw "Key=Value" tag strings into a dict.

        A tag without an "=" is kept verbatim, under itself as both key and
        value, rather than dropped
        """
        parsed: Dict[str, str] = {}
        for tag in tags:
            key, sep, value = tag.partition("=")
            if sep:
                parsed[key] = value
            else:
                parsed[tag] = tag
        return parsed
