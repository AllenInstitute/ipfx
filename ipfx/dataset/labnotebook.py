import h5py
import math
from ipfx.py2to3 import to_str

class LabNotebookReader(object):
    """
        The Lab Notebook Reader class
        
        This class reads two sections: numeric data and text data
    """
    
    def __init__(self):
        self.register_enabled_names()

    def register_enabled_names(self):
        """
            register_enabled_names:
            mapping of notebook keys to keys representing if that value is
            enabled move this to subclasses if/when key names diverge
        """
        self.enabled = {}
        self.enabled["V-Clamp Holding Level"] = "V-Clamp Holding Enable"
        self.enabled["RsComp Bandwidth"] = "RsComp Enable"
        self.enabled["RsComp Correction"] = "RsComp Enable"
        self.enabled["RsComp Prediction"] = "RsComp Enable"
        self.enabled["Whole Cell Comp Cap"] = "Whole Cell Comp Enable"
        self.enabled["Whole Cell Comp Resist"] = "Whole Cell Comp Enable"
        self.enabled["I-Clamp Holding Level"] = "I-Clamp Holding Enable"
        self.enabled["Neut Cap Value"] = "Neut Cap Enable"
        self.enabled["Bridge Bal Value"] = "Bridge Bal Enable"

    def get_numeric_value(self, name, data_col, sweep_col, enable_col, sweep_num, default_val):
        """
            fetch numeric data

            val_number has 3 dimensions -- the first has a shape of
            (#fields * 9). there are many hundreds of elements in this
            dimension. they look to represent the full array of values
            (for each field for each multipatch) for a given point in
            time, and thus given sweep
            according to Thomas Braun (igor nwb dev), the first 8 pages are
            for headstage data, and the 9th is for headstage-independent
            data

            Parameters
            ----------
            name: str
            data_col: int
            sweep_col: int
            enable_col: int
            sweep_num: int
            default_val: dict

            Returns
            -------
            
            last non-empty entry in specified column
            for specified sweep number
        """
        data = self.val_number
        
        return_val = default_val
        for sample in data:
            swp = sample[sweep_col][0]
            if math.isnan(swp):
                continue
            if int(swp) == sweep_num:
                if enable_col is not None and sample[enable_col][0] != 1.0:
                    continue # 'enable' flag present and it's turned off
                val = sample[data_col][0]
                if not math.isnan(val):
                    return_val = val
        return return_val

    def get_text_value(self, name, data_col, sweep_col, enable_col, sweep_num, default_val):
        """
            fetch text data

            Parameters
            ----------
            name: str
            data_col: int
            sweep_col: int
            enable_col: int
            sweep_num: int
            default_val: dict

            Returns
            -------
            
            last non-empty entry in specified column
            for specified sweep number
        """

        data = self.val_text
        
        return_val = default_val
        for sample in data:
            swp = sample[sweep_col][0]
            if len(swp) == 0:
                continue
            if int(swp) == int(sweep_num):
                if enable_col is not None:
                    print("Error: Enable flag not expected for text values")
                val = sample[data_col][0]
                if len(val) > 0:
                    return_val = val
        return return_val

    def get_value(self, name, sweep_num, default_val):
        """
            looks for key in lab notebook and returns the value associated with
            the specified sweep, or the default value if no value is found
            (NaN and empty strings are considered to be non-values)

            name_number has 3 dimensions -- the first has shape
            (#fields * 9) and stores the key names. the second looks
            to store units for those keys. The third is numeric text
            but it's role isn't clear

            val_number has 3 dimensions -- the first has a shape of
            (#fields * 9). there are many hundreds of elements in this
            dimension. they look to represent the full array of values
            (for each field for each multipatch) for a given point in
            time, and thus given sweep

            Parameters
            ----------
            name: str
            sweep_num: int
            default_val: dict

            Returns
            -------
            values obtained from lab notebook
            
        """

        numeric_fields = [to_str(c) for c in self.colname_number[0]]
        text_fields = [to_str(c) for c in self.colname_text[0]]
        
        if name in numeric_fields:
            sweep_idx = numeric_fields.index("SweepNum")
            enable_idx = None
            if name in self.enabled:
                enable_col = self.enabled[name]
                enable_idx = numeric_fields.index(enable_col)
            field_idx = numeric_fields.index(name)
            return self.get_numeric_value(name, field_idx, sweep_idx, enable_idx, sweep_num, default_val)
        elif name in text_fields:
            if "Sweep #" in text_fields:
                sweep_idx = text_fields.index("Sweep #")
            else:
                sweep_idx = text_fields.index("SweepNum")
            enable_idx = None
            if name in self.enabled:
                enable_col = self.enabled[name]
                enable_idx = text_fields.index(enable_col)
            field_idx = text_fields.index(name)
            return self.get_text_value(name, field_idx, sweep_idx, enable_idx, sweep_num, default_val)
        else:
            return default_val

class LabNotebookReaderIgorNwb(LabNotebookReader):
    """ 
        LabNotebookReaderIgorNwb:
        Loads lab notebook data out of an Igor-generated NWB file.
        Module input is the name of the nwb file.
        Notebook data can be read through get_value() function
    """
    def __init__(self, nwb_file):
        LabNotebookReader.__init__(self)

        h5 = h5py.File(nwb_file, "r")

        # check NWB version and whether labnotebook exist
        for k in h5["general/labnotebook"]:
            notebook = h5["general/labnotebook"][k]
            break
        # load column data into memory
        self.val_text = notebook["textualValues"][()]
        self.colname_text = notebook["textualKeys"][()]
        self.val_number = notebook["numericalValues"][()]
        self.colname_number = notebook["numericalKeys"][()]
        h5.close()
        
        self.register_enabled_names()

    def get_textual_values(self):
        return self.val_text

    def get_textual_keys(self):
        return self.colname_text

    def get_numerical_values(self):
        return self.val_number

    def get_numerical_keys(self):
        return self.colname_number
