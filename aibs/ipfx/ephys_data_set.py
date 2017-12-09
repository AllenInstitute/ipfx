
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
    
    def __init__(self):
        self.sweep_table = None

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
            mask = st[self.STIMULUS_CODE].apply(self.stimulus_code_matches, args=stimulus_codes)
            st = st[mask]

        return st

    @staticmethod
    def stimulus_code_matches(stimulus_code, match_codes):
        return any(stimulus_code.startswith(mc) for mc in match_codes)

    def sweep(self, sweep_number):
        """ returns a dictionary """
        raise NotImplementedError

    def aligned_sweep(self, sweep_number, stim_onset_delta):
        sweep = self.get_sweep(sweep_number)

        # do it
        
        
