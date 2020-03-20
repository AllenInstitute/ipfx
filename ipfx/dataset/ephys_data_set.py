from typing import Optional, List, Dict, Any, Tuple
import warnings
import logging
import pandas as pd
import numpy as np
from ipfx.sweep import Sweep, SweepSet
from ipfx.dataset.ephys_data_interface import EphysDataInterface


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

    def __init__(self,
                 data: EphysDataInterface,
                 sweep_info: Optional[Dict[str, Any]] = None):

        self.data = data
        self.ontology = data.ontology

        if sweep_info is None:
            sweep_info = self.extract_sweep_info()

        self.build_sweep_table(sweep_info)

    def extract_sweep_info(self) -> Dict[str,Any]:
        """
        Extract sweep information for all sweeps

        Returns
        -------
        sweep information
        """

        sweep_info = []

        for index, sweep_map in self.data.sweep_map_table.iterrows():
            sweep_num = sweep_map["sweep_number"]
            sweep_info.append(self.data.extract_sweep_record(sweep_num))

        return sweep_info

    def build_sweep_table(self, sweep_info=None):
        """
        Construct pd.Dataframe from
        Parameters
        ----------
        sweep_info

        Returns
        -------

        """
        if sweep_info:
            self.sweep_table = pd.DataFrame.from_records(sweep_info)
        else:
            self.sweep_table = pd.DataFrame(columns=self.COLUMN_NAMES)

    def filtered_sweep_table(self,
                             clamp_mode=None,
                             stimuli=None,
                             stimuli_exclude=None,
                             ):

        st = self.sweep_table

        if clamp_mode:
            mask = st[self.CLAMP_MODE] == clamp_mode
            st = st[mask.astype(bool)]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, args=(stimuli,), tag_type="code")
            st = st[mask.astype(bool)]

        if stimuli_exclude:
            mask = ~st[self.STIMULUS_CODE].apply(
                self.ontology.stimulus_has_any_tags, args=(stimuli_exclude,), tag_type="code")
            st = st[mask.astype(bool)]

        return st

    def get_sweep_number(self, stimuli, clamp_mode=None):

        sweeps = self.filtered_sweep_table(clamp_mode=clamp_mode,
                                           stimuli=stimuli).sort_values(by=self.SWEEP_NUMBER)

        if len(sweeps) > 1:
            logging.warning(
                "Found multiple sweeps for stimulus %s: using largest sweep number" % str(stimuli))

        if len(sweeps) == 0:
            raise IndexError("Cannot find {} sweeps with clamp mode: {} found ".format(stimuli,clamp_mode))

        return sweeps.sweep_number.values[-1]

    def voltage_current(self, sweep_data, clamp_mode) -> Tuple[np.arrray,np.array]:

        if clamp_mode == self.VOLTAGE_CLAMP:
            v = sweep_data['stimulus']
            i = sweep_data['response']
        elif clamp_mode == self.CURRENT_CLAMP:
            v = sweep_data['response']
            i = sweep_data['stimulus']
        else:
            raise Exception(f"Invalid clamp mode: {clamp_mode}")

        return v, i

    def sweep(self, sweep_number):

        """
        Create an instance of the Sweep class with the data loaded from the from a file

        Parameters
        ----------
        sweep_number: int

        Returns
        -------
        sweep: Sweep object
        """

        sweep_data = self.data.get_sweep_data(sweep_number)
        self.drop_trailing_zeros_in_response(sweep_data)
        sweep_record = self.data.get_sweep_record(sweep_number)
        sampling_rate = sweep_data['sampling_rate']
        dt = 1. / sampling_rate
        t = np.arange(0, len(sweep_data['stimulus'])) * dt

        epochs = sweep_data.get('epochs')
        clamp_mode = sweep_record['clamp_mode']

        v, i = self.voltage_current(sweep_data,clamp_mode)

        v *= 1.0e3    # convert units V->mV
        i *= 1.0e12   # convert units A->pA

        if len(sweep_data['stimulus']) != len(sweep_data['response']):
            warnings.warn("Stimulus duration {} is not equal reponse duration {}".
                          format(len(sweep_data['stimulus']),len(sweep_data['response'])))

        try:
            sweep = Sweep(t=t,
                          v=v,
                          i=i,
                          sampling_rate=sampling_rate,
                          sweep_number=sweep_number,
                          clamp_mode=clamp_mode,
                          epochs=epochs,
                          )

        except Exception:
            logging.warning("Error reading sweep %d" % sweep_number)
            raise

        return sweep

    def sweep_set(self, sweep_numbers):
        try:
            return SweepSet([self.sweep(sn) for sn in sweep_numbers])
        except TypeError:  # not iterable
            return SweepSet([self.sweep(sweep_numbers)])

    @staticmethod
    def drop_trailing_zeros_in_response(sweep_data):

        response = sweep_data['response']

        if len(np.flatnonzero(response)) == 0:
            recording_end_idx = 0
            sweep_end_idx = 0
        else:
            recording_end_idx = np.flatnonzero(response)[-1]
            sweep_end_idx = len(response)-1

        if recording_end_idx < sweep_end_idx:
            response[recording_end_idx+1:] = np.nan
            sweep_data["response"] = response


