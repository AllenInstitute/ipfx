import json

class EphysStimulusOntology(object):
    def __init__(self, file_name):
        with open(file_name,'r') as f:
            self.mapping = json.load(f)


class EphysDataSet(object):
    STIMULUS_NAME = 'stimulus_name'
    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
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

        self.ramp_codes = ( "C1RP", )
        self.ramp_names = ( "Ramp", )

        self.long_square_names = ( "Long Square", )
        self.coarse_long_square_codes = ( "C1LSCOARSE", )
        self.long_square_codes = ( "C1LS", )

        self.short_square_triple_names = ( "Short Square - Triple", ) 
        
        self.short_square_names = ( "Short Square",
                                    "Short Square - Hold -60mv",
                                    "Short Square - Hold -70mv",
                                    "Short Square - Hold -80mv" )

        self.blowout_codes = ( 'EXTPBLWOUT', )
        self.bath_codes = ( 'EXTPINBATH', )
        self.seal_codes = ( 'EXTPCllATT', )
        self.breakin_codes = ( 'EXTPBREAKN', )
        self.extp_codes = ( 'EXTP', )

        self.current_clamp_units = ( 'Amps', 'pA' )

    def filtered_sweep_table(self, current_clamp_only=False, passing_only=False, 
                             stimulus_names=None, stimulus_codes=None):
        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(self.current_clamp_units)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimulus_names:
            st = st[st[self.STIMULUS_NAME].isin(stimulus_names)]

        if stimulus_codes:
            mask = st[self.STIMULUS_CODE].apply(self.stimulus_code_matches, args=(stimulus_codes,))
            st = st[mask]

        return st

    @staticmethod
    def stimulus_code_matches(stimulus_code, match_codes):
        return any(stimulus_code.startswith(mc) for mc in match_codes)

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
