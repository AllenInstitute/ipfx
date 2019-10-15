import numpy as np
from ipfx.ephys_data_set import Sweep, SweepSet
from collections import defaultdict

class MPSweep(Sweep):
    """Adapter for neuroanalysis.Recording => ipfx.Sweep
    """
    def __init__(self, rec, test_pulse=True):
        pri = rec['primary']
        cmd = rec['command']
        t = pri.time_values
        v = pri.data * 1e3  # convert to mV
        # TODO: select holding item explicitly; don't assume it is [0]
        holding = rec.stimulus.items[0].amplitude  
        # convert to pA with holding current removed
        i = (cmd.data - holding) * 1e12   
        srate = pri.sample_rate
        sweep_num = rec.parent.key
        clamp_mode = rec.clamp_mode  # this will be 'ic' or 'vc'; not sure if that's right

        Sweep.__init__(self, t, v, i,
                       clamp_mode=clamp_mode,
                       sampling_rate=srate,
                       sweep_number=sweep_num,
                       epochs=None,
                       test_pulse=test_pulse)


stim_list = [
    'TargetV_DA_0',
    'If_Curve_DA_0',
    # 'Chirp_DA_0',
    # 'TargetV_DA_0'
]

def sweeps_dict_from_cell(cell):
    recordings = cell.electrode.recordings
    sweeps_dict = {stim:list() for stim in stim_list}
    for recording in recordings:
        for name in stim_list:
            if recording.patch_clamp_recording.stim_name == name:
                sweeps_dict[name].append(recording.sync_rec.ext_id)
    return sweeps_dict

def mpsweep_duration(mpsweep):
    if mpsweep.select_epoch('stim'):
        duration = mpsweep.t[-1] - mpsweep.t[0]
        mpsweep.select_epoch('sweep')
        return duration
    else:
        return None

def min_duration_of_sweeplist(sweep_list):
    if len(sweep_list)==0:
        return 0
    else:
        return min(mpsweep_duration(mpsweep) for mpsweep in sweep_list)

def mp_cell_id(cell):
    """Get an id for an MP database cell object (combined timestamp and cell id).
    """
    cell_id = "{ts:0.3f}_{ext_id}".format(ts=cell.experiment.acq_timestamp, ext_id=cell.ext_id)
    return cell_id

def cell_from_mpid(mpid):
    """Get an MP database cell object by its id (combined timestamp and cell id).
    """
    import multipatch_analysis.database as db
    timestamp, ext_id = mpid.split('_')
    timestamp = float(timestamp)
    ext_id = int(ext_id)
    experiment = db.experiment_from_timestamp(timestamp)
    cell = experiment.cells[ext_id]
    return cell

def mpsweep_from_recording(recording):
    """Get an MPSweep object containing sweep data from a MP database recording object.
    """
    electrode = recording.electrode
    miesnwb = electrode.experiment.data
    sweep_id = recording.sync_rec.ext_id
    sweep = miesnwb.contents[sweep_id][electrode.device_id]
    return MPSweep(sweep)

def mp_project_cell_ids(project, max_count=None, filter_cells=True):
    """Get a list of ids for a multipatch project.
    """
    import multipatch_analysis.database as db
    session = db.get_default_session()
    if filter_cells:
        q_base = session.query(db.Cell).join(db.Electrode).join(db.Recording).join(db.PatchClampRecording).filter(
            db.PatchClampRecording.qc_pass==True)
        q1 = q_base.filter(db.PatchClampRecording.stim_name=='TargetV_DA_0').group_by(db.Cell.id)
        q2 = q_base.filter(db.PatchClampRecording.stim_name=='If_Curve_DA_0').group_by(db.Cell.id)
        q_cells = q1.intersect(q2)
    else:
        q_cells = session.query(db.Cell)

    q = q_cells.join(db.Experiment).filter(db.Experiment.project_name==project)
    if max_count:
        cells = q.slice(0, max_count)
    else:
        cells = q.all()
    return [mp_cell_id(cell) for cell in cells]
    