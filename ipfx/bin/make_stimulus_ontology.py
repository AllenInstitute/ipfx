import allensdk.core.json_utilities as ju
from ipfx.stimulus import StimulusOntology
import re
import ipfx.lims_queries as lq
import logging


def make_stimulus_ontology(stims):

    NAME = 'name'
    CODE = 'code'
    RES = 'resolution'
    CORE = 'core'
    HOLD = 'hold'

    ontology_tags = []
    stims = [{k: v for k, v in stim.items()} for stim in stims]

    for stim in stims:
        tags = set()

        sname = stim['stimulus_name']
        scode = stim['stimulus_code']

        # code tags
        m = re.search("(.*)\d{6}$", scode)
        if m:
            code_name, = m.groups()
            tags.add((CODE, code_name, scode))
        else:
            tags.add((CODE, scode))

        # core tags
        if scode.startswith('C1'):
            tags.add((CORE, 'Core 1'))
        elif scode.startswith('C2'):
            tags.add((CORE, 'Core 2'))

        # resolution tags
        if 'FINE' in scode:
            tags.add((RES, 'Fine'))
        elif 'COARSE' in scode:
            tags.add((RES, 'Coarse'))

        # name tags
        if 'C1NS' in scode:
            tags.add((NAME, 'Noise', sname))
        elif 'Short Square' in sname and 'Triple' not in sname:
            tags.add((NAME, 'Short Square'))
        elif 'Long Square' in sname:
            tags.add((NAME, 'Long Square'))
        else:
            tags.add((NAME, sname))

        # hold tags
        if 'Hold' in sname:
            # find the first dash
            idx = sname.find('-')
            b = sname[idx + 1:]
            tags.add((HOLD, b.strip()))

        ontology_tags.append(list(tags))

    return ontology_tags


def make_stimulus_ontology_from_lims(file_name):

    if lq.able_to_connect_to_lims():
        stims = lq.get_stimuli_description()
        stim_ontology = make_stimulus_ontology(stims)
        ju.write(file_name, stim_ontology)
        logging.info("Updated stimulus ontology from LIMS")


def make_default_stimulus_ontology():

    stimulus_ontology_file = StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE
    make_stimulus_ontology_from_lims(stimulus_ontology_file)


def main():

    make_default_stimulus_ontology()


if __name__== "__main__": 
    main()
