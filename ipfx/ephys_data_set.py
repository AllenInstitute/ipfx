import warnings
import logging
import pandas as pd
import numpy as np

from ipfx.sweep import Sweep, SweepSet
from ipfx.stimulus import StimulusOntology

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

    def __init__(
            self, 
            ontology, 
            validate_stim=True, 
            deprecation_warning=False
    ):

        self.sweep_table = None
        self.ontology = ontology or StimulusOntology.default()
        self.validate_stim = validate_stim

        if deprecation_warning:
            warnings.warn(np.VisibleDeprecationWarning((
                "Instead of constructing {} instances "
                "directly, use ipfx.data_set_utils.create_data_set"
                "this will make it easier to transition to ipfx 1.0.0"
            ).format(type(self))))

    @property
    def nwb_data(self):
        warnings.warn(np.VisibleDeprecationWarning(
                "In ipfx 1.0.0 nwb_data will not be a public attribute of "
                "EphysDataSet"
        ))
        return self._nwb_data

    
    def build_sweep_table(self, sweep_info=None, deprecation_warning=True):

        if deprecation_warning:
            warnings.warn(np.VisibleDeprecationWarning(
                "in ipfx version 1.0.0 build_sweep_table will not be a "
                "public method of EphysDataSet"
            ))

        if sweep_info:
            self.add_clamp_mode(
                sweep_info, deprecation_warning=deprecation_warning
            )
            self.sweep_table = pd.DataFrame.from_records(sweep_info)
        else:
            self.sweep_table = pd.DataFrame(columns=self.COLUMN_NAMES)

    def add_clamp_mode(self, sweep_info, deprecation_warning=True):
        """
        Check if clamp mode is available and otherwise detect it
        Parameters
        ----------
        sweep_info

        Returns
        -------

        """

        if deprecation_warning:
            warnings.warn(np.VisibleDeprecationWarning(
                "in ipfx version 1.0.0 add_clamp_mode will not be a "
                "public method of EphysDataSet"
            ))

        for sweep_record in sweep_info:
            sweep_number = sweep_record["sweep_number"]
            sweep_record[self.CLAMP_MODE] = self.get_clamp_mode(sweep_number)

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

    def get_sweep_record(self, sweep_number):
        """
        Parameters
        ----------
        sweep_number: int sweep number

        Returns
        -------
        sweep_record: dict of sweep properties
        """

        st = self.sweep_table

        if sweep_number is not None:
            mask = st[self.SWEEP_NUMBER] == sweep_number
            st = st[mask]

        return st.to_dict(orient='records')[0]

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

        sweep_data = self.get_sweep_data(sweep_number)
        sweep_record = self.get_sweep_record(sweep_number)
        sampling_rate = sweep_data['sampling_rate']
        dt = 1. / sampling_rate
        t = np.arange(0, len(sweep_data['stimulus'])) * dt

        epochs = sweep_data.get('epochs')
        clamp_mode = sweep_record['clamp_mode']

        if clamp_mode == "VoltageClamp":
            v = sweep_data['stimulus']
            i = sweep_data['response']
        elif clamp_mode == "CurrentClamp":
            v = sweep_data['response']
            i = sweep_data['stimulus']
        else:
            raise Exception("Unable to determine clamp mode for sweep " + sweep_number)

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

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        raise NotImplementedError

    def extract_sweep_stim_info(self):
        """
        Returns
        -------
        sweep_props: list of dicts
            where each dict includes sweep properties
        """

        raise NotImplementedError

    def get_recording_date(self):
        raise NotImplementedError

    def get_sweep_data(self, sweep_number):
        """
        Read sweep data from the nwb file
        Substitute trailing zeros in the response for np.nan
        because trailing zeros occur after the end of recording

        Parameters
        ----------
        sweep_number

        Returns
        -------
        dict in the format:

        {
            'stimulus': np.ndarray,
            'response': np.ndarray,
            'stimulus_unit': string,
            'sampling_rate': float
        }
        """

        sweep_data = self._nwb_data.get_sweep_data(sweep_number)

        response = sweep_data['response']

        nonzero =  np.flatnonzero(response)
        if len(nonzero) == 0:
            recording_end_idx = 0
        else:
            recording_end_idx = nonzero[-1] + 1

        sweep_data["response"] = response[:recording_end_idx]
        sweep_data["stimulus"] = sweep_data["stimulus"][:recording_end_idx]

        return sweep_data

    def get_clamp_mode(self,stimulus_number):
        raise NotImplementedError

    def get_stimulus_code(self,stimulus_number):
        raise NotImplementedError

    def get_stimulus_name(self, stim_code):

        if not self.ontology:
            raise ValueError("Missing stimulus ontology")

        try:
            stim = self.ontology.find_one(stim_code, tag_type="code")
            return stim.tags(tag_type="name")[0][-1]

        except KeyError:
            if self.validate_stim:
                raise
            else:
                warnings.warn("Stimulus code {} is not in the ontology".format(stim_code))
                return
