from typing import List, Dict, Optional

from ipfx.qc_feature_extractor import sweep_qc_features
import ipfx.sweep_props as sweep_props
import ipfx.qc_feature_evaluator as qcp
from ipfx.stimulus import StimulusOntology
from ipfx.dataset.ephys_data_set import EphysDataSet

from pynwb.icephys import SweepTable
import pynwb


def drop_failed_sweeps(
        dataset: EphysDataSet,
        stimulus_ontology: Optional[StimulusOntology] = None,
        qc_criteria: Optional[Dict] = None
) -> List[Dict]:
    """A convenience which extracts and QCs sweeps in preparation for dataset
    feature extraction. This function:
    1. extracts sweep qc features
    2. removes sweeps tagged with failure messages
    3. sets sweep states based on qc results

    Parameters
    ----------
    dataset : dataset from which to draw sweeps

    Returns
    -------
    sweep_features : a list of dictionaries, each describing a sweep
    """
    if stimulus_ontology is None:
        stimulus_ontology = StimulusOntology.default()
    if qc_criteria is None:
        qc_criteria = qcp.load_default_qc_criteria()

    sweep_features = sweep_qc_features(dataset)
    sweep_props.drop_tagged_sweeps(sweep_features)
    sweep_props.remove_sweep_feature("tags", sweep_features)
    sweep_states = qcp.qc_sweeps(
        stimulus_ontology, sweep_features, qc_criteria
    )
    sweep_props.assign_sweep_states(sweep_states, sweep_features)

    dataset.sweep_info = sweep_features


def inject_sweep_table(nwbfile: pynwb.NWBFile):
    """
    Allows us to keep using the SweepTable which can not be constructed anymore in pynwb 3.0.
    """

    sweep_table = SweepTable.__new__(SweepTable, parent=nwbfile, in_construct_mode=True)
    sweep_table.__init__(name='sweep_table')
    sweep_table._in_construct_mode = False
    nwbfile.sweep_table = sweep_table
