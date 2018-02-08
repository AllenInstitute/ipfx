import allensdk.internal.core.lims_utilities as lu
import allensdk.core.json_utilities as ju
import re

stims = lu.query("""
select ersn.name as stimulus_code, est.name as stimulus_name from ephys_raw_stimulus_names ersn
join ephys_stimulus_types est on ersn.ephys_stimulus_type_id = est.id
""")

out = []

for stim in stims:
    tags = set()

    sname = stim['stimulus_name']
    scode = stim['stimulus_code']

    tags.add(scode)

    m = re.search("(.*)\d{6}$", scode)
    if m:
        code_name, = m.groups()
        tags.add(code_name)

    tags.add(sname)
    
    if scode.startswith('C1'):
        tags.add('Core 1')
    elif scode.startswith('C2'):
        tags.add('Core 2')

    if 'C1NS' in scode:
        tags.add('Noise')

    if 'FINE' in scode:
        tags.add('Fine')

    if 'COARSE' in scode:
        tags.add('Coarse')
    
    if 'Short Square' in sname and 'Triple' not in sname:
        tags.add('Short Square')

    if 'Long Square' in sname:
        tags.add('Long Square')

    if 'Hold' in sname:
        # find the first dash
        idx = sname.find('-')
        a = sname[:idx]
        b = sname[idx+1:]
        tags.add(a.strip())
        tags.add(b.strip())

    out.append({ 'code': scode, 'name': sname, 'tags': list(tags) })

out.append({ 'code': 'C1NSSEED', 'name': 'Noise 1', 'tags': [ 'Noise', 'Core 1'] })
ju.write('stimulus_ontology.json', out)
