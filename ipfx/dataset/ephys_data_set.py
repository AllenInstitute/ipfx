from typing import (
    Optional, List, Dict, Tuple, Collection, Sequence, Union
)
import logging
from collections import defaultdict
import copy as cp

import pandas as pd
import numpy as np

from allensdk.deprecated import deprecated

from ipfx.dataset.ephys_data_interface import EphysDataInterface
from ipfx.stimulus import StimulusOntology
from ipfx.sweep import Sweep, SweepSet


class EphysDataSet(object):

    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    STIMULUS_AMPLITUDE = 'stimulus_amplitude'
    STIMULUS_NAME = 'stimulus_name'
    SWEEP_NUMBER = 'sweep_number'
    CLAMP_MODE = 'clamp_mode'

    COLUMN_NAMES = [STIMULUS_UNITS,
                    STIMULUS_CODE,
                    STIMULUS_AMPLITUDE,
                    STIMULUS_NAME,
                    CLAMP_MODE,
                    SWEEP_NUMBER,
                    ]

    LONG_SQUARE = 'long_square'
    COARSE_LONG_SQUARE = 'coarse_long_square'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'
    SHORT_SQUARE = 'short_square'
    RAMP = 'ramp'

    VOLTAGE_CLAMP = "VoltageClamp"
    CURRENT_CLAMP = "CurrentClamp"

    @property
    def ontology(self) -> StimulusOntology:
        """The stimulus ontology maps codified description of the stimulus type 
        to the human-readable descriptions.
        """
        return self._data.ontology

    @property
    def sweep_table(self) -> pd.DataFrame:
        """Each row of the sweep table contains the metadata for a single 
        sweep. In particular details of the stimulus presented and the clamp 
        mode. See EphysDataInterface.get_sweep_metadata for more information.

        """
        if not hasattr(self, "_sweep_table"):
            sweeps: List[Dict] = []
            for num in self._data.sweep_numbers:
                current = self._data.get_sweep_metadata(num)

                if self._sweep_info:
                    info = self._sweep_info.get(num, None)
                    if info is None:
                        continue
                    current.update(info)
                sweeps.append(current)

            self._sweep_table = pd.DataFrame(sweeps)
        return self._sweep_table

    @property
    def sweep_info(self):
        return list(self._sweep_info.values())

    @sweep_info.setter
    def sweep_info(self, value):
        if not isinstance(value, dict):
            self._sweep_info: Dict = {}
            for sweep in value:
                self._sweep_info[sweep["sweep_number"]] = sweep
        else:
            self._sweep_info = value
        
        if hasattr(self, "_sweep_table"):
            del self._sweep_table

    def __init__(
            self,
            data: EphysDataInterface,
            sweep_info: Optional[List[Dict]] = None
    ):
        """EphysDataSet is the preferred interface for running analyses or 
        pipeline code.

        Parameters
        ----------
        data : This object must implement the EphysDataInterface. It will 
            handle any loading of data from external sources (such as NWB2 
            files)
        """
        self._data: EphysDataInterface = data
        self.sweep_info = sweep_info or []

    def _setup_stimulus_repeat_lookup(self):
        """Each sweep contains the ith repetition of some stimulus (from 1 -> 
        the number of times that stimulus was presented). Find i for each 
        sweep.

        Notes
        -----
        see get_stim_code_ext for use

        """
        stimulus_counters = defaultdict(int)
        self._stimulus_repeat_lookup = {}

        for sweep_number in self._data.sweep_numbers:
            code = self.get_stimulus_code(sweep_number)
            stimulus_counters[code] += 1
            self._stimulus_repeat_lookup[sweep_number] = \
                stimulus_counters[code]

    def filtered_sweep_table(
            self,
            clamp_mode: Optional[str] = None,
            stimuli: Optional[Collection[str]] = None,
            stimuli_exclude: Optional[Collection[str]] = None,
    ) -> pd.DataFrame:
        """Utility for filtering the sweep table

        Parameters
        ----------
        clamp_mode: filter to one of self.VOLTAGE_CLAMP or self.CURRENT_CLAMP
        stimuli: filter to sweeps presenting these stimuli (codes)
        stimuli_exclude: filter to sweeps not presenting these stimuli

        Returns
        -------
        filtered sweep table
        """
        st = self.sweep_table

        if clamp_mode:
            mask = st[self.CLAMP_MODE] == clamp_mode
            st = st[mask.astype(bool)]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, 
                args=(stimuli,), 
                tag_type="code"
            )
            st = st[mask.astype(bool)]

        if stimuli_exclude:
            mask = ~st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, 
                args=(stimuli_exclude,), 
                tag_type="code"
            )
            st = st[mask.astype(bool)]

        return st

    def get_sweep_numbers(
            self,
            stimuli: Collection[str] = None,
            clamp_mode: Optional[str] = None
    ) -> List[int]:
        """Return the integer identifier of all sweeps matching argued criteria

        Parameters
        ----------
        stimuli : filter to  sweeps presenting these stimuli (codes)
        clamp_mode : filter to sweeps of this clamp mode

        Returns
        -------
        A list of sweep numbers matching these criteria
        """

        sweeps = self.filtered_sweep_table(
            clamp_mode=clamp_mode, stimuli=stimuli
        ).sort_values(by=self.SWEEP_NUMBER)

        if len(sweeps) == 0:
            raise IndexError(
                f"Cannot find {stimuli} sweeps with clamp mode: {clamp_mode} "
            )

        return sweeps[self.SWEEP_NUMBER].values.tolist()

    @deprecated("call .get_sweep_numbers()[-1] instead")
    def get_sweep_number(
            self,
            stimuli: Collection[str],
            clamp_mode: Optional[str] = None
    ) -> int:
        """Convenience for getting the integer identifier of the temporally 
        latest sweep matching argued criteria.

        Parameters
        ----------
        stimuli : filter to  sweeps presentingthese stimuli
        clamp_mode : filter to sweeps of this clamp mode

        Returns
        -------
        The identifier of the last sweep matching argued criteria
        """
        return self.get_sweep_numbers(stimuli, clamp_mode)[-1]

    def sweep(self, sweep_number: int) -> Sweep:
        """
        Create an instance of the Sweep class with the data loaded from the 
        from a file

        Parameters
        ----------
        sweep_number: int

        Returns
        -------
        sweep: Sweep object
        """

        sweep_data = self.get_sweep_data(sweep_number)
        sweep_metadata = self._data.get_sweep_metadata(sweep_number)

        time = np.arange(
            len(sweep_data["stimulus"])
        ) / sweep_data["sampling_rate"]

        voltage, current = type(self)._voltage_current(
            sweep_data["stimulus"],
            sweep_data["response"], 
            sweep_metadata["clamp_mode"], 
            enforce_equal_length=True,
        )

        try:
            sweep = Sweep(
                t=time,
                v=voltage,
                i=current,
                sampling_rate=sweep_data["sampling_rate"],
                sweep_number=sweep_number,
                clamp_mode=sweep_metadata["clamp_mode"],
                epochs=sweep_data.get("epochs", None),
            )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep

    def sweep_set(
            self, 
            sweep_numbers: Union[Sequence[int], int, None] = None
    ) -> SweepSet:
        """Construct a SweepSet object, which offers convenient access to an 
        ordered collection of sweeps.

        Parameters
        ----------
        sweep_numbers : Identifiers for the sweeps which will make up this set. 
            If None, use all available sweeps.

        Returns
        -------
        A SweepSet constructed from the requested sweeps
        """

        if sweep_numbers is None:
            _sweep_numbers: Sequence = self._data.sweep_numbers
        elif not hasattr(sweep_numbers, "__len__"):  # not testing for order
            _sweep_numbers = [sweep_numbers]
        else:
            _sweep_numbers = sweep_numbers  # type: ignore

        return SweepSet([self.sweep(num) for num in _sweep_numbers])

    def get_recording_date(self) -> str:
        """Return the date and time at which recording began.

        Returns
        -------
        a string, formatted like: "%Y-%m-%d %H:%M:%S" in local time
        """
        return (
            self._data.get_full_recording_date()
                .strftime("%Y-%m-%d %H:%M:%S")
        )

    def get_sweep_data(self, sweep_number: int) -> Dict:
        """Obtain the recorded data for a given sweep.

        Parameters
        ----------
        sweep_number : identifier for the sweep whose data will be returned

        Returns
        -------
        A dictionary containing at least:
            {
                'stimulus': np.ndarray,
                'response': np.ndarray,
                'stimulus_unit': string,
                'sampling_rate': float
            }
        """
        sweep_data = cp.copy(self._data.get_sweep_data(sweep_number))

        response = sweep_data['response']

        nonzero = np.flatnonzero(response)
        if len(nonzero) == 0:
            recording_end_idx = 0
        else:
            recording_end_idx = nonzero[-1] + 1

        sweep_data["response"] = response[:recording_end_idx]
        sweep_data["stimulus"] = sweep_data["stimulus"][:recording_end_idx]

        return sweep_data

    def get_clamp_mode(self, sweep_number: int) -> str:
        """Obtain the clamp mode of a given sweep. Should be one of 
        EphysDataSet.VOLTAGE_CLAMP or EphysDataSet.CURRENT_CLAMP

        Parameters
        ----------
        sweep_number : identifier for the sweep whose clamp mode will be 
            returned

        Returns
        -------
        The clamp mode of the identified sweep
        """
        return self._data.get_sweep_metadata(sweep_number)["clamp_mode"]

    def get_stimulus_code(self, sweep_number: int) -> str:
        """Return the (short form) stimulus code for a particular sweep.

        Parameters
        ----------
        sweep_number : identifier for the sweep whose stimulus code will be 
            returned

        Returns
        -------
        code defining the stimulus presented on the identified sweep
        """
        return self._data.get_stimulus_code(sweep_number)

    def get_stimulus_code_ext(self, sweep_number: int) -> str:
        """Obtain the extended stimulus code for a sweep. This is the stimulus 
        code for that sweep augmented with an integer counter describing the 
        number of presentations of that stimulus up to and including the 
        requested sweep.

        Parameters
        ----------
        sweep_number : identifies the sweep whose extended stimulus code will 
        be returned

        Returns
        -------
        A string of the form "{stimulus_code}[{counter}]"
        """
        if not hasattr(self, "self._stimulus_repeat_lookup"):
            self._setup_stimulus_repeat_lookup()

        repeat = self._stimulus_repeat_lookup[sweep_number]
        code = self.get_stimulus_code(sweep_number)
        return f"{code}[{repeat}]"

    def get_stimulus_units(self, sweep_number: int) -> str:
        """Report the SI unit of measurement for a sweep's stimulus data

        Parameters
        ----------
        sweep_number : identifies the sweep whose stimulus unit will be 
            returned

        Returns
        -------
        An SI (or derived) unit's name
        """
        return self._data.get_sweep_metadata(sweep_number)["stimulus_units"]

    @classmethod
    def _voltage_current(
            cls,
            stimulus: np.ndarray,
            response: np.ndarray, 
            clamp_mode: str,
            enforce_equal_length: bool = True
    ) -> Tuple[np.array, np.array]:
        """Resolve the stimulus and response arrays from a sweep's data into 
        voltage and current, using the clamp mode as a guide

        Parameters
        ----------
        stimulus : stimulus trace
        response : response trace
        clamp_mode : Used to map stimulus and response to voltage and current
        enforce_equal_length : Raise a ValueError if the stimulus and 
            response arrays have uneven numbers of samples

        Returns
        -------
        The voltage and current traces.

        """            

        if clamp_mode == cls.VOLTAGE_CLAMP:
            voltage = stimulus
            current = response
        elif clamp_mode == cls.CURRENT_CLAMP:
            voltage = response
            current = stimulus
        else:
            raise ValueError(f"Invalid clamp mode: {clamp_mode}")

        if enforce_equal_length and len(voltage) != len(current):
            raise ValueError(
                f"found {len(voltage)} voltage samples, "
                f"but {len(current)} current samples"
            )

        return voltage, current
