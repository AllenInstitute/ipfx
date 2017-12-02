
class EphysDataSet(object):
    def get_sweep_table(self):
        """ returns a dataframe """
        raise NotImplementedError

    def get_sweep(self, sweep_number):
        """ returns a dictionary """
        raise NotImplementedError
