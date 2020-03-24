import os
import logging
import pg8000

from allensdk.core.authentication import credential_injector
from allensdk.core.auth_config import LIMS_DB_CREDENTIAL_MAP

from ipfx.py2to3 import to_str


TIMEOUT = os.environ.get(
    "IPFX_LIMS_TIMEOUT",
    os.environ.get(
        "IPFX_TEST_TIMEOUT",
        None
    )
)
if TIMEOUT is not None:
    TIMEOUT = float(TIMEOUT)  # type: ignore


@credential_injector(LIMS_DB_CREDENTIAL_MAP)
def _connect(user, host, dbname, password, port, timeout=TIMEOUT):

    conn = pg8000.connect(
        user=user, 
        host=host, 
        database=dbname, 
        password=password, 
        port=int(port),
        timeout=timeout
    )
    return conn, conn.cursor()


def able_to_connect_to_lims():

    try:
        conn, cursor = _connect()
        cursor.close()
        conn.close()
    except pg8000.Error:
        # the connection failed
        return False
    except TypeError:
        # a credential was missing
        return False

    return True


def _select(cursor, query, parameters=None):
    if parameters is None:
        cursor.execute(query)
    else:
        pg8000.paramstyle = 'numeric'
        cursor.execute(query, parameters)
    columns = [ to_str(d[0]) for d in cursor.description ]
    return [ dict(zip(columns, c)) for c in cursor.fetchall() ]


def query(query, parameters=None):
    conn, cursor = _connect()
    try:
        results = _select(cursor, query, parameters=parameters)
    finally:
        cursor.close()
        conn.close()
    return results


def get_input_nwb_file(specimen_id):

    sql="""
    select err.storage_directory||'EPHYS_FEATURE_EXTRACTION_V2_QUEUE_'||err.id||'_input.json' as input_v2_json,
           err.storage_directory||'EPHYS_FEATURE_EXTRACTION_QUEUE_'||err.id||'_input.json' as input_v1_json,
           err.storage_directory||err.id||'.nwb' as nwb_file
    from specimens sp
    join ephys_roi_results err on err.id = sp.ephys_roi_result_id
    where sp.id = %d
    """ % specimen_id
    res = query(sql)[0]
    res = { k:v for k,v in res.items() }

    # if the input_v2_json does not exist, then use input_v1_json instead:
    if os.path.isfile(res["input_v2_json"]):
        res["input_json"] = res["input_v2_json"]
    else:
        res["input_json"] = res["input_v1_json"]

    nwb_file_name  = res['nwb_file']

    return nwb_file_name


def get_input_h5_file(specimen_id):

    h5_res = query("""
    select err.*, wkf.*,sp.name as specimen_name
    from ephys_roi_results err
    join specimens sp on sp.ephys_roi_result_id = err.id
    join well_known_files wkf on wkf.attachable_id = err.id
    where sp.id = %d
    and wkf.well_known_file_type_id = 306905526
    """ % specimen_id)

    h5_file_name = os.path.join(h5_res[0]['storage_directory'], h5_res[0]['filename']) if len(h5_res) else None

    return h5_file_name


def get_sweep_states(specimen_id):

    sweep_states = []

    res = query("""
        select sweep_number, workflow_state from ephys_sweeps
        where specimen_id = %d
        """ % specimen_id)

    for sweep in res:
        # only care about manual calls
        if sweep["workflow_state"] == "manual_passed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': True})
        elif sweep["workflow_state"] == "manual_failed":
            sweep_states.append({'sweep_number': sweep["sweep_number"],
                                 'passed': False})

    return sweep_states


def get_stimuli_description():

    stims = query("""
    select ersn.name as stimulus_code, est.name as stimulus_name from ephys_raw_stimulus_names ersn
    join ephys_stimulus_types est on ersn.ephys_stimulus_type_id = est.id
    """)

    return stims


def get_specimen_info_from_lims_by_id(specimen_id):

    result = query("""
                  SELECT s.name, s.ephys_roi_result_id, s.id
                  FROM specimens s
                  WHERE s.id = %s
                  """ % specimen_id)
    if len(result) == 0:
        logging.info("No result from query to find specimen information")
        return None, None, None

    result = result[0]

    if result:
        return result["name"], result["ephys_roi_result_id"], result["id"]
    else:
        logging.info("Could not find specimen {:d}".format(specimen_id))
        return None, None, None


def get_nwb_path_from_lims(ephys_roi_result):
    """
    Try to find NWBIgor file preferentially
    If not found, look for a processed NWB file

    well known file type ID for NWB files is 475137571
    well known file type ID for NWBIgor files is 570280085


    Parameters
    ----------
    ephys_roi_result: int

    Returns
    -------
    full path of the nwb file

    """

    result = query("""
    SELECT f.filename, f.storage_directory FROM well_known_files f
    WHERE f.attachable_type = 'EphysRoiResult' AND f.attachable_id = %s AND f.well_known_file_type_id = 570280085
    """ % (ephys_roi_result,))

    if len(result) == 0:
        logging.warning("Fall back to looking for NWB type")

        result = query("""
        SELECT f.filename, f.storage_directory FROM well_known_files f
        WHERE f.attachable_type = 'EphysRoiResult' AND f.attachable_id = %s AND f.well_known_file_type_id = 475137571
        """ % (ephys_roi_result,))

    result = result[0]

    if result:
        nwb_path = result["storage_directory"] + result["filename"]
        return nwb_path
    else:
        logging.info("Cannot find NWB file")
        return None


def get_igorh5_path_from_lims(ephys_roi_result):

    sql = """
    SELECT f.filename, f.storage_directory
    FROM well_known_files f
    WHERE f.attachable_type = 'EphysRoiResult'
    AND f.attachable_id = %s
    AND f.well_known_file_type_id = 306905526
    """ % ephys_roi_result

    result = query(sql)
    if len(result) == 0:
        logging.info("No result from query to find Igor H5 file")
        return None

    result = result[0]

    if result:
        h5_path = result["storage_directory"] + result["filename"]
        return h5_path
    else:
        logging.info("Cannot find Igor H5 file")
        return None


def project_specimen_ids(project, passed_only=True):

    SQL = """
        SELECT sp.id FROM specimens sp
        JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
        JOIN projects prj ON prj.id = sp.project_id
        WHERE prj.code = '%s'
        """ % project

    if passed_only:
        SQL += "AND err.workflow_state = 'manual_passed'"

    results = query(SQL)
    sp_ids = [d["id"] for d in results]
    return sp_ids
