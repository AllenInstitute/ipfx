from __future__ import print_function
import numpy as np
from ipfx.aibs_data_set import AibsDataSet
import ipfx.data_set_features as dsf
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.feature_vectors as fv
import pytest

@pytest.mark.skip
def test_first_ap_waveform():
    pass


@pytest.mark.skip
def test_isi_shape():

    nwb_file = "/allen/programs/celltypes/production/humancelltypes/prod79/Ephys_Roi_Result_737293859/H18.03.315.11.11.01.05.nwb"
    data_set = AibsDataSet(nwb_file= nwb_file)
    ontology = data_set.ontology

    lsq_sweep_numbers = data_set.filtered_sweep_table(clamp_mode=data_set.CURRENT_CLAMP,
                                                      stimuli=ontology.long_square_names).sweep_number.sort_values().values

    lsq_sweeps = data_set.sweep_set(lsq_sweep_numbers)
    print(lsq_sweep_numbers)
    lsq_start, lsq_dur, _, _, _ = stf.get_stim_characteristics(lsq_sweeps.sweeps[0].i,
                                                               lsq_sweeps.sweeps[0].t)

    lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(lsq_sweeps,
                                                  start=lsq_start,
                                                  end=lsq_start + lsq_dur,
                                                  **dsf.detection_parameters(data_set.LONG_SQUARE))
    lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-100.)

    lsq_features = lsq_an.analyze(lsq_sweeps)


    # Figure out the sampling rate & target length
    swp = lsq_sweeps.sweeps[0]

    sampling_rate = int(np.rint(1. / (swp.t[1] - swp.t[0])))
    window_length = 0.003
    length_in_points = int(sampling_rate * window_length)

# Long squares
    if lsq_sweeps is not None and lsq_features is not None:
        rheo_ind = lsq_features["rheobase_sweep"].name

        sweep = lsq_sweeps.sweeps[rheo_ind]
        spikes = lsq_features["spikes_set"][rheo_ind]
        print(rheo_ind, fv.first_ap_waveform(sweep, spikes, length_in_points))


    assert 0

