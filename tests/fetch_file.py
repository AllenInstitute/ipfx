import os
import urllib2
import shutil
from allensdk.api.queries.cell_types_api import CellTypesApi


def get_celltypes_file(folder, specimen_id):
    nwb_file = '{}.nwb'.format(specimen_id)

    file_path = os.path.join(folder, nwb_file)

    if not os.path.exists(file_path):
        ct = CellTypesApi()
        ct.save_ephys_data(specimen_id, file_path)

    return file_path


def fetch_test_file(folder, nwb_file):
    file_path = os.path.join(folder, nwb_file)

    if not os.path.exists(file_path):

        BASE_URL = "https://www.byte-physics.de/Downloads/allensdk-test-data/"

        response = urllib2.urlopen(BASE_URL + nwb_file)
        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(response, out_file)

    return file_path
