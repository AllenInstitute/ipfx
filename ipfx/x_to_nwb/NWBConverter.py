"""
Convert NWB v1 to NWB v2 files.
"""

import logging


import numpy as np

from pynwb import NWBHDF5IO, NWBFile
from pynwb.icephys import CurrentClampStimulusSeries, VoltageClampStimulusSeries
from pynwb.icephys import CurrentClampSeries, VoltageClampSeries
from ipfx.py2to3 import to_str


import ipfx.lab_notebook_reader as lab_notebook_reader
import ipfx.nwb_reader as nwb_reader


log = logging.getLogger(__name__)


class NWBConverter:

    V_CLAMP_MODE = "voltage_clamp"
    I_CLAMP_MODE = "current_clamp"
    PLACEHOLDER = "PLACEHOLDER"

    def __init__(self, input_file, output_file):
        """
        Convert NWB v1 to v2

        """

        self.nwb_data = nwb_reader.create_nwb_reader(input_file)
        self.notebook = lab_notebook_reader.create_lab_notebook_reader(input_file)

        nwb_file = self.create_nwb_file()

        device = nwb_file.create_device(name='electrode_0')

        electrode = nwb_file.create_ic_electrode(name="elec0",
                                                 description=' some kind of electrode',
                                                 device=device)

        for i in self.create_stimulus_series(electrode):
            nwb_file.add_stimulus(i)

        for i in self.create_acquisition_series(electrode):
            nwb_file.add_acquisition(i)

        with NWBHDF5IO(output_file, "w") as io:
            io.write(nwb_file, cache_spec=True)



    def create_nwb_file(self):
        """
        Create a pynwb NWBFile object from the nwb v1 file contents.
        """


        session_description = self.PLACEHOLDER
        identifier = self.PLACEHOLDER
        session_start_time = self.nwb_data.get_session_start_time()
        experiment_description = "IVSCC ephys recording"
        session_id = self.PLACEHOLDER


        return NWBFile(session_description=session_description,
                       identifier=identifier,
                       session_start_time=session_start_time,
                       experimenter=None,
                       experiment_description=experiment_description,
                       session_id=session_id)

    def get_clamp_mode(self, sweep_num):

        attrs = self.nwb_data.get_sweep_attrs(sweep_num)
        ancestry = attrs["ancestry"]

        time_series_type = to_str(ancestry[-1])
        if "CurrentClamp" in time_series_type:
            clamp_mode = self.I_CLAMP_MODE
        elif "VoltageClamp" in time_series_type:
            clamp_mode = self.V_CLAMP_MODE
        else:
            raise Exception("Unable to determine clamp mode in {}".format(sweep_num))

        return clamp_mode


    def get_stimulus_series_class(self,clamp_mode):
        """
        Return the appropriate pynwb stimulus class for the given clamp mode.
        """

        if clamp_mode == self.V_CLAMP_MODE:
            return VoltageClampStimulusSeries
        elif clamp_mode == self.I_CLAMP_MODE:
            return CurrentClampStimulusSeries
        else:
            raise ValueError(f"Unsupported clamp mode {clamp_mode}.")


    def get_stim_code_ext(self, stim_code, sweep_num):
        cnt = self.notebook.get_value("Set Sweep Count", sweep_num, 0)
        stim_code_ext = stim_code + "[%d]" % int(cnt)

        return stim_code_ext


    def create_stimulus_series(self,electrode):

        series = []

        for sweep_name in self.nwb_data.get_sweep_names():

            sweep_number = self.nwb_data.get_sweep_number(sweep_name)
            stimulus = self.nwb_data.get_stimulus(sweep_number)
            clamp_mode = self.get_clamp_mode(sweep_number)
            stim_code = self.nwb_data.get_stim_code(sweep_number)
            stim_code_ext = self.get_stim_code_ext(stim_code,sweep_number)
            scale_factor = self.notebook.get_value("Scale Factor", sweep_number, None)

            StimulusSeries = self.get_stimulus_series_class(clamp_mode)
            stimulus_series = StimulusSeries(name=sweep_name,
                                             data=stimulus['data'],
                                             sweep_number=sweep_number,
                                             unit=stimulus['unit'],
                                             electrode=electrode,
                                             gain=scale_factor,
                                             resolution=np.nan,
                                             conversion=stimulus['conversion'],
                                             starting_time=np.nan,
                                             rate=stimulus['rate'],
                                             comments=stimulus['comment'],
                                             stimulus_description=stim_code_ext)

            series.append(stimulus_series)

        return series

    def create_acquisition_series(self, electrode):

        series = []

        for sweep_name in self.nwb_data.get_sweep_names():

            sweep_number = self.nwb_data.get_sweep_number(sweep_name)
            acquisition = self.nwb_data.get_acquisition(sweep_number)
            stim_code = self.nwb_data.get_stim_code(sweep_number)
            stim_code_ext = self.get_stim_code_ext(stim_code,sweep_number)

            scale_factor = self.notebook.get_value("Scale Factor", sweep_number, None)

            clamp_mode = self.get_clamp_mode(sweep_number)
            if clamp_mode == self.V_CLAMP_MODE:
                acquisition_series = VoltageClampSeries(name=sweep_name,
                                                        sweep_number=sweep_number,
                                                        data=acquisition['data'],
                                                        unit=acquisition['unit'],
                                                        conversion=acquisition['conversion'],
                                                        resolution=np.nan,
                                                        starting_time=np.nan,
                                                        rate=acquisition['rate'],
                                                        electrode=electrode,
                                                        gain=scale_factor,
                                                        capacitance_slow=np.nan,
                                                        resistance_comp_correction=np.nan,
                                                        capacitance_fast=np.nan,
                                                        resistance_comp_bandwidth=np.nan,
                                                        resistance_comp_prediction=np.nan,
                                                        comments=acquisition['comment'],
                                                        whole_cell_capacitance_comp=np.nan,
                                                        whole_cell_series_resistance_comp=np.nan,
                                                        stimulus_description=stim_code_ext
                                                 )
            elif clamp_mode == self.I_CLAMP_MODE:

                bridge_balance = self.notebook.get_value("Bridge Bal Value", sweep_number, np.nan)
                bias_current = self.notebook.get_value("I-Clamp Holding Level", sweep_number, np.nan)
                acquisition_series = CurrentClampSeries(name=sweep_name,
                                                        sweep_number=sweep_number,
                                                        data=acquisition['data'],
                                                        unit=acquisition['unit'],
                                                        conversion=acquisition['conversion'],
                                                        resolution=np.nan,
                                                        starting_time=np.nan,
                                                        rate=acquisition['rate'],
                                                        comments=acquisition['comment'],
                                                        electrode=electrode,
                                                        gain=scale_factor,
                                                        bias_current=bias_current,
                                                        bridge_balance=bridge_balance,
                                                        stimulus_description=stim_code_ext,
                                                        capacitance_compensation=np.nan
                )

            series.append(acquisition_series)

        return series
