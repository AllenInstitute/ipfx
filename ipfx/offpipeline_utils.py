import numpy as np
import pandas as pd
import logging
import traceback
from functools import partial
from multiprocessing import Pool
import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
from ipfx.aibs_data_set import AibsDataSet
from ipfx.bin.run_feature_vector_extraction import lims_nwb_information, sdk_nwb_information

def dataset_for_specimen_id(specimen_id, data_source="lims", ontology=None):
    """Construct an AibsDataSet for a cell by specimen_id
    
    Parameters
    ----------
    specimen_id : int or str
    data_source : str, optional
        "lims" or "sdk", by default "lims"
    ontology : StimulusOntology, optional
    
    Returns
    -------
    AibsDataSet
    """
    ontology = ontology or StimulusOntology(ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE))
    logging.debug("specimen_id: {}".format(specimen_id))
    # Find or retrieve NWB file and ancillary info and construct an AibsDataSet object
    if data_source == "lims":
        nwb_path, h5_path = lims_nwb_information(specimen_id)
        if type(nwb_path) is dict and "error" in nwb_path:
            error_dict = nwb_path
            raise IOError("Problem getting NWB file for specimen {:d} from LIMS".format(specimen_id))

        dataset = AibsDataSet(
                nwb_file=nwb_path, h5_file=h5_path, ontology=ontology)
    elif data_source == "sdk":
        nwb_path, sweep_info = sdk_nwb_information(specimen_id)
        dataset = AibsDataSet(
                nwb_file=nwb_path, sweep_info=sweep_info, ontology=ontology)
    else:
        raise ValueError("invalid data source specified ({})".format(data_source))

    return dataset

def sweepset_by_type_qc(dataset, specimen_id, stimuli_names=None, sweep_qc_option="none"):
    sweep_numbers = sweep_numbers_by_type_qc(dataset, specimen_id, 
                        stimuli_names=stimuli_names, sweep_qc_option=sweep_qc_option)
    return dataset.sweep_set(sweep_numbers)

def sweep_numbers_by_type_qc(dataset, specimen_id, stimuli_names=None, sweep_qc_option="none"):
    exist_sql = """
        select swp.sweep_number from ephys_sweeps swp
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
    """

    passed_sql = """
        select swp.sweep_number from ephys_sweeps swp
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
        and swp.workflow_state like '%%passed'
    """

    passed_except_delta_vm_sql = """
        select swp.sweep_number, tag.name
        from ephys_sweeps swp
        join ephys_sweep_tags_ephys_sweeps estes on estes.ephys_sweep_id = swp.id
        join ephys_sweep_tags tag on tag.id = estes.ephys_sweep_tag_id
        where swp.specimen_id = :1
        and swp.sweep_number = any(:2)
    """

    iclamp_st = dataset.filtered_sweep_table(clamp_mode=dataset.CURRENT_CLAMP, stimuli=stimuli_names)

    if sweep_qc_option == "none":
        return iclamp_st["sweep_number"].sort_values().values
    else:
        # check that sweeps exist in LIMS
        sweep_num_list = iclamp_st["sweep_number"].sort_values().tolist()
        results = lq.query(exist_sql, (specimen_id, sweep_num_list))
        res_nums = pd.DataFrame(results, columns=["sweep_number"])["sweep_number"].tolist()

        not_checked_list = []
        for swp_num in sweep_num_list:
            if swp_num not in res_nums:
                logging.debug("Could not find sweep {:d} from specimen {:d} in LIMS for QC check".format(swp_num, specimen_id))
                not_checked_list.append(swp_num)

        # get straight-up passed sweeps
        results = lq.query(passed_sql, (specimen_id, sweep_num_list))
        results_df = pd.DataFrame(results, columns=["sweep_number"])
        passed_sweep_nums = results_df["sweep_number"].values

        if sweep_qc_option == "lims-passed-only":
            return np.sort(np.hstack([passed_sweep_nums, np.array(not_checked_list)])) # deciding to keep non-checked sweeps for now
        elif sweep_qc_option == "lims-passed-except-delta-vm":

            # also get sweeps that only fail due to delta Vm
            failed_sweep_list = list(set(sweep_num_list) - set(passed_sweep_nums))
            if len(failed_sweep_list) == 0:
                return np.sort(passed_sweep_nums)
            results = lq.query(passed_except_delta_vm_sql, (specimen_id, failed_sweep_list))
            results_df = pd.DataFrame(results, columns=["sweep_number", "name"])

            # not all cells have tagged QC status - if there are no tags assume the
            # fail call is correct and exclude those sweeps
            tagged_mask = np.array([sn in results_df["sweep_number"].tolist() for sn in failed_sweep_list])

            # otherwise, check for having an error tag that isn't 'Vm delta'
            # and exclude those sweeps
            has_non_delta_tags = np.array([np.any((results_df["sweep_number"].values == sn) &
                (results_df["name"].values != "Vm delta")) for sn in failed_sweep_list])

            also_passing_nums = np.array(failed_sweep_list)[tagged_mask & ~has_non_delta_tags]

            return np.sort(np.hstack([passed_sweep_nums, also_passing_nums, np.array(not_checked_list)]))
        else:
            raise ValueError("Invalid sweep-level QC option {}".format(sweep_qc_option))

def run_feature_collection(feature_fcn, ids=None, project="T301", sweep_qc_option='none', include_failed_cells=False,
        output_file=None, n_procs=0, **kwargs):
    if ids is not None:
        specimen_ids = ids
    else:
        specimen_ids = lq.project_specimen_ids(project, passed_only=not include_failed_cells)

    logging.info("Number of specimens to process: {:d}".format(len(specimen_ids)))

    get_data_partial = partial(feature_fcn, sweep_qc_option=sweep_qc_option, **kwargs)

    if n_procs:
        pool = Pool(n_procs)
        results = pool.map(get_data_partial, specimen_ids)
    else:
        results = map(get_data_partial, specimen_ids)

    df = pd.DataFrame.from_records(results, index=specimen_ids)
    if output_file:
        df.to_csv(output_file)
    return df