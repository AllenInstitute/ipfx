import allensdk.internal.core.lims_utilities as lu
import allensdk.core.json_utilities as ju
import re

stims = lu.query("""
select ersn.name as stimulus_code, est.name as stimulus_name from ephys_raw_stimulus_names ersn
join ephys_stimulus_types est on ersn.ephys_stimulus_type_id = est.id
""")

out = []

# all core one sweeps
# ont.has_any(stim, ('Core 1'))

# only Coarse Long Squares
# ont.has_all(stim, ('Long Square', 'Coarse'))

#
#
stims =  [ { k.decode("UTF-8"):v for k,v in stim.items() } for stim in stims   ]

NAME = 'name'
CODE = 'code'
RES = 'resolution'
CORE = 'core'
HOLD = 'hold'

for stim in stims:
    tags = set()

    sname = stim['stimulus_name']
    scode = stim['stimulus_code']

    #tags.add(scode)

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
        a = sname[:idx]
        b = sname[idx+1:]
        #tags.add(a.strip())
        #tags.add(b.strip())
        tags.add((HOLD, b.strip()))

    out.append(list(tags))

out.append([ (CODE, 'C1NSSEED'), (NAME, 'Noise', 'Noise 1'), (CORE, 'Core 1') ])

for o in out:
    print(o)

ju.write('stimulus_ontology_patchseq.json', out)
