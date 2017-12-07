
class EphysDataSet(object):
    STIMULUS_NAME = 'stimulus_name'
    STIMULUS_UNITS = 'stimulus_units'
    STIMULUS_CODE = 'stimulus_code'
    SWEEP_NUMBER = 'sweep_number'
    PASSED = 'passed'

    CURRENT_CLAMP_UNITS = [ 'Amps', 'pA' ]
    
    def __init__(self):
        self.sweep_table = None
    
    def filtered_sweep_table(self, current_clamp_only, passing_only, 
                             stimulus_names=None, stimulus_code_prefixes=None):
        st = self.sweep_table

        if current_clamp_only:
            st = st[st[self.STIMULUS_UNITS].isin(self.CURRENT_CLAMP_UNITS)]

        if passing_only:
            st = st[st[self.PASSED]]

        if stimulus_names:
            st = st[st[self.STIMULUS_NAME].isin(stimulus_names)]

        if stimulus_code_prefixes:
            st = st[st[self.STIMULUS_CODE].str.startswith(stimulus_code_prefixes)]

        return st

    def sweep(self, sweep_number):
        """ returns a dictionary """
        raise NotImplementedError

    def aligned_sweep(self, sweep_number, stim_onset_delta):
        sweep = self.get_sweep(sweep_number)

        # do it
        
        
