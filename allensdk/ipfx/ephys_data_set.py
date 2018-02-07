class EphysStimulusOntology(object):
    def __init__(self, stimuli):
        self.stimuli = stimuli
        
    def find(self, val, key='code'):
        try:
            return next(s for s in self.stimuli if s[key] == val)
        except StopIteration as e:
            raise KeyError("Could not find stimulus: %s" % val)

    def matches_any(self, query_stim, tags, key='code'):
        query_stim = self.find(query_stim, key)

        return any(tag in query_stim['tags'] for tag in tags)

    def matches_all(self, query_stim, tags, key='code'):
        query_stim = self.find(query_stim, key)

        return all(tag in query_stim['tags'] for tag in tags)


class EphysDataSet(object):
    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    STIMULUS_NAME = 'stimulus_name'
    SWEEP_NUMBER = 'sweep_number'
    PASSED = 'passed'
  
    LONG_SQUARE = 'long_square'
    COARSE_LONG_SQUARE = 'coarse_long_square'
    SHORT_SQUARE_TRIPLE = 'short_square_triple'
    SHORT_SQUARE= 'short_square'
    RAMP = 'ramp'
    
    def __init__(self, ontology=None):
        self.sweep_table = None
        self.ontology = ontology

        self.ramp_names = ( "Ramp", )

        self.long_square_names = ( "Long Square", )
        self.coarse_long_square_names = ( "C1LSCOARSE",  )
        self.short_square_triple_names = ( "Short Square - Triple", ) 
        
        self.short_square_names = ( "Short Square",
                                    "Short Square - Hold -60mV",
                                    "Short Square - Hold -70mV",
                                    "Short Square - Hold -80mV" )

        self.blowout_names = ( 'EXTPBLWOUT', )
        self.bath_names = ( 'EXTPINBATH', )
        self.seal_names = ( 'EXTPCllATT', )
        self.breakin_names = ( 'EXTPBREAKN', )
        self.extp_names = ( 'EXTP', )


        self.current_clamp_units = ( 'Amps', 'pA') 

    def filtered_sweep_table(self, current_clamp_only=False, passing_only=False, stimuli=None):
        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(self.current_clamp_units)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimuli:
            mask = st[self.STIMULUS_CODE].apply(self.ontology.matches_any, args=(stimuli,))
            st = st[mask]

        return st

    def sweep(self, sweep_number):
        """ returns a dictionary with properties: i (in pA), v (in mV), t (in sec), start, end"""
        raise NotImplementedError

    def sweep_set(self, sweep_numbers):
        return SweepSet([ self.sweep(sn) for sn in sweep_numbers ])

    def aligned_sweeps(self, sweep_numbers, stim_onset_delta):
        pass

class Sweep(object):
    def __init__(self, t, v, i, expt_start=None, expt_end=None, sampling_rate=None):
        self.t = t
        self.v = v
        self.i = i

        self.expt_start = expt_start if expt_start else 0
        self.expt_end = expt_end if expt_end else self.t_end
        self.sampling_rate = sampling_rate
        
    @property
    def t_end(self):
        return self.t[-1]
    

class SweepSet(object):
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def _prop(self, prop):
        return [ getattr(s, prop) for s in self.sweeps ]
 
    @property
    def t(self):
        return self._prop('t')

    @property
    def v(self):
        return self._prop('v')

    @property
    def i(self):
        return self._prop('i')

    @property 
    def expt_start(self):
        return self._prop('expt_start')
    
    @property
    def expt_end(self):
        return self._prop('expt_end')

    @property
    def t_end(self):
        return self._prop('t_end')
