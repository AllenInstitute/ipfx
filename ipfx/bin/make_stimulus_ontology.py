import allensdk.core.json_utilities as ju
import re
import os
import lims_queries as lq

NAME = 'name'
CODE = 'code'
RES = 'resolution'
CORE = 'core'
HOLD = 'hold'


def make_stimulus_ontology(stims):

    ontology = []

    stims = [{k.decode("UTF-8"): v for k, v in stim.items()} for stim in stims]

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

        ontology.append(list(tags))

    return ontology


def make_stimulus_ontology_from_lims():

    stimuli = lq.get_stimuli_description()

    if not stimuli:
        return None

    ontology = make_stimulus_ontology(stimuli)

    ontology.append([(CODE, 'C1NSSEED'), (NAME, 'Noise', 'Noise 1'), (CORE, 'Core 1')])

    return ontology


def make_default_stimulus_ontology():

    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    PACKAGE_DIR = os.path.dirname(MODULE_DIR)

    stim_ontology_tags = make_stimulus_ontology_from_lims()

    if not stim_ontology_tags:
        return

    stim_ontology_json = os.path.join(PACKAGE_DIR,"defaults/stimulus_ontology.json")

    ju.write(stim_ontology_json, stim_ontology_tags)

    for o in stim_ontology_tags:
        print(o)


def main():

    make_default_stimulus_ontology()


if __name__=="__main__": main()
