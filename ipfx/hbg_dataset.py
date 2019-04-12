import pandas as pd
import numpy as np
import logging
import ipfx.nwb_reader as nwb_reader

from .ephys_data_set import EphysDataSet


class HBGDataSet(EphysDataSet):
    def __init__(self, sweep_info=None, nwb_file=None, ontology=None, api_sweeps=True, validate_stim=True):
        super(HBGDataSet, self).__init__(ontology, validate_stim)
        self.nwb_data = nwb_reader.create_nwb_reader(nwb_file)

        if sweep_info is not None:
            sweep_info = self.modify_api_sweep_info(
                sweep_info) if api_sweeps else sweep_info
        else:
            sweep_info = self.extract_sweep_meta_data()

        self.sweep_table = pd.DataFrame.from_records(sweep_info)

    def extract_sweep_meta_data(self):

        logging.debug("Build sweep table")

        sweep_props = []

        def get_finite_or_none(d, key):

            try:
                value = d[key]
            except KeyError:
                return None

            if np.isnan(value):
                return None

            return value

        for index, sweep_map in self.nwb_data.sweep_map_table.iterrows():
            sweep_record = {}
            sweep_num = sweep_map["sweep_number"]
            attrs = self.nwb_data.get_sweep_attrs(sweep_num)
            sweep_record["sweep_number"] = sweep_num

            timeSeriesType = attrs["neurodata_type"]

            if "CurrentClamp" in timeSeriesType:
                sweep_record["stimulus_units"] = "A"
                sweep_record["clamp_mode"] = "CurrentClamp"
            elif "VoltageClamp" in timeSeriesType:
                sweep_record["stimulus_units"] = "V"
                sweep_record["clamp_mode"] = "VoltageClamp"
            else:
                raise ValueError("Unexpected TimeSeries type {}.".format(timeSeriesType))

            sweep_record["bridge_balance_mohm"] = get_finite_or_none(attrs, "bridge_balance")
            sweep_record["leak_pa"] = get_finite_or_none(attrs, "bias_current")
            sweep_record["stimulus_scale_factor"] = get_finite_or_none(attrs, "gain")

            sweep_record["stimulus_code_ext"] = None  # not required anymore
            sweep_record["stimulus_code"] = self.nwb_data.get_stim_code(sweep_num)

            if self.ontology:
                sweep_record["stimulus_name"] = self.get_stimulus_name(sweep_record["stimulus_code"])

            sweep_props.append(sweep_record)

        return sweep_props

    def get_sweep_data(self, sweep_number):
        return self.nwb_data.get_sweep_data(sweep_number)
