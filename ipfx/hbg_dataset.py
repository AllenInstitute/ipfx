import pandas as pd
import numpy as np
import logging
import warnings
import ipfx.nwb_reader as nwb_reader

from .ephys_data_set import EphysDataSet


class HBGDataSet(EphysDataSet):
    def __init__(
            self, 
            sweep_info=None, 
            nwb_file=None, 
            ontology=None, 
            api_sweeps=True, 
            validate_stim=True,
            deprecation_warning=True
    ):
        super(HBGDataSet, self).__init__(
            ontology, validate_stim, deprecation_warning=deprecation_warning
        )
        self._nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_info is None:
            sweep_info = self.extract_sweep_stim_info()

        self.build_sweep_table(sweep_info, deprecation_warning=False)


    def extract_sweep_stim_info(self):

        logging.debug("Build sweep table")

        sweep_info = []

        def get_finite_or_none(d, key):

            try:
                value = d[key]
            except KeyError:
                return None

            if np.isnan(value):
                return None

            return value

        for index, sweep_map in self._nwb_data.sweep_map_table.iterrows():
            sweep_record = {}
            sweep_num = sweep_map["sweep_number"]
            sweep_record["sweep_number"] = sweep_num

            attrs = self._nwb_data.get_sweep_attrs(sweep_num)

            sweep_record["stimulus_units"] = self.get_stimulus_units(sweep_num)

            sweep_record["bridge_balance_mohm"] = get_finite_or_none(attrs, "bridge_balance")
            sweep_record["leak_pa"] = get_finite_or_none(attrs, "bias_current")
            sweep_record["stimulus_scale_factor"] = get_finite_or_none(attrs, "gain")

            sweep_record["stimulus_code"] = self.get_stimulus_code(sweep_num)
            sweep_record["stimulus_code_ext"] = self.get_stimulus_code_ext(sweep_num)

            if self.ontology:
                sweep_record["stimulus_name"] = self.get_stimulus_name(sweep_record["stimulus_code"])

            sweep_info.append(sweep_record)

        return sweep_info

    def get_stimulus_units(self, sweep_num):

        unit_str = self._nwb_data.get_stimulus_unit(sweep_num)
        return unit_str

    def get_clamp_mode(self, sweep_num):

        attrs = self._nwb_data.get_sweep_attrs(sweep_num)
        timeSeriesType = attrs["neurodata_type"]

        if "CurrentClamp" in timeSeriesType:
            clamp_mode = self.CURRENT_CLAMP
        elif "VoltageClamp" in timeSeriesType:
            clamp_mode = self.VOLTAGE_CLAMP
        else:
            raise ValueError("Unexpected TimeSeries type {}.".format(timeSeriesType))

        return clamp_mode

    def get_stimulus_code(self, sweep_num):

        stim_code_ext = self._nwb_data.get_stim_code(sweep_num)

        return stim_code_ext.split("[")[0]

    def get_stimulus_code_ext(self, sweep_num):

        return self._nwb_data.get_stim_code(sweep_num)

    def get_recording_date(self):
        return self._nwb_data.get_recording_date()
