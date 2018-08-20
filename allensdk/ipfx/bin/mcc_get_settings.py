#!/usr/bin/env python

"""
Code for interfacing with the Multi Clamp Commander application from Axon/Molecular Devices
"""

import ctypes as ct
import os
import json
import datetime

# Original code taken from https://github.com/tgbugs/inferno/core/mcc.py, commit 3d555888 (Update README.md, 2017-08-02)
# Original License: MIT
#
# Code heavily modified by Thomas Braun

API_VERSION_STR = b'1.0.0.9'
V_CLAMP_MODE = 0
I_CLAMP_MODE = 1

# clamp mode defintion
MCC_MODE_DICT = {0: 'VC',
                 1: 'IC',
                 2: 'IEZ'}

c_int_p = ct.POINTER(ct.c_int)
c_uint_p = ct.POINTER(ct.c_uint)
c_bool_p = ct.POINTER(ct.c_bool)
c_double_p = ct.POINTER(ct.c_double)
c_string_p = ct.POINTER(ct.c_char)


class DataGatherer():
    """
    Collect data from all available MCC amplifiers
    """

    def __init__(self):
        pass

    def getData(self, mcc):
        """
        Return all MCC settings for all amplifiers as dictionary
        """

        settings = {}

        for uid in mcc.getUIDs():
            mcc.selectUniqueID(uid)

            settings[uid] = {}

            for name in self._getListOfFunctions(mcc.GetMode()):
                func = getattr(mcc, name)

                settings[uid][name] = func()

        return settings

    def _getListOfFunctions(self, clampMode):
        """
        Return a list of MultiClampControl getter functions with settings for
        the current amplifier and the current clamp mode.
        """

        VCfuncs = ["GetHoldingEnable",
                   "GetHolding",
                   "GetOscKillerEnable",
                   "GetRsCompBandwidth",
                   "GetRsCompCorrection",
                   "GetRsCompEnable",
                   "GetRsCompPrediction",
                   "GetWholeCellCompEnable",
                   "GetWholeCellCompCap",
                   "GetWholeCellCompResist",
                   "GetFastCompCap",
                   "GetSlowCompCap",
                   "GetFastCompTau",
                   "GetSlowCompTau",
                   "GetSlowCompTauX20Enable"]

        ICfuncs = ["GetHoldingEnable",
                   "GetHolding",
                   "GetNeutralizationEnable",
                   "GetNeutralizationCap",
                   "GetBridgeBalEnable",
                   "GetBridgeBalResist",
                   "GetSlowCurrentInjEnable",
                   "GetSlowCurrentInjLevel",
                   "GetSlowCurrentInjSettlingTime"]

        funcs = ["GetMode",
                 "GetModeSwitchEnable",
                 "GetPipetteOffset",
                 "GetPrimarySignal",
                 "GetPrimarySignalGain",
                 "GetPrimarySignalLPF",
                 "GetPrimarySignalHPF",
                 "GetScopeSignalLPF",
                 "GetSecondarySignal",
                 "GetSecondarySignalGain",
                 "GetSecondarySignalLPF",
                 "GetOutputZeroEnable",
                 "GetOutputZeroEnable",
                 "GetLeakSubEnable",
                 "GetLeakSubResist",
                 "GetPulseAmplitude",
                 "GetPulseDuration",
                 "GetZapDuration",
                 "GetBuzzDuration",
                 "GetMeterResistEnable",
                 "GetMeterIrmsEnable"]

        if clampMode == V_CLAMP_MODE:
            return VCfuncs + funcs
        elif clampMode == I_CLAMP_MODE:
            return ICfuncs + funcs
        else:
            raise RuntimeError(f"Unexpected clamp mode {clampMode}")


# utility function for type casting and returning the value of a pointer
def val(ptr, ptype):
    if ptype == ct.c_char_p:
        return ct.cast(ptr, ptype).value
    return ct.cast(ptr, ptype)[0]


class MultiClampControl:
    """
    Class for interacting with the MultiClamp Commander from Axon/Molecular Devices

    Usage:

    ```
      import mcc

      m = mcc.MultiClampControl()
      UIDs = m.getUIDs()  # output all found amplifiers
      m.selectUniqueID(next(iter(UIDs)))  # select the first one (implicitly done by __init__)
      clampMode = m.GetMode() # return the clamp mode
      if m._handleError()[0]:
        # handle error
        pass
    ```
    """

    def __init__(self, dllPath=None):

        dllPaths = [("C:/Program Files/Molecular Devices/MultiClamp 700B Commander/"
                     "3rd Party Support/AxMultiClampMsg/"),
                    ("C:/Program Files (x86)/Molecular Devices/MultiClamp 700B Commander/"
                     "3rd Party Support/AxMultiClampMsg/")]

        if dllPath:
            dllPaths.insert(0, dllPath)

        for path in dllPaths:
            if os.path.isdir(path):
                self._getDLL(path)

        self._pnError = ct.byref(ct.c_int())

        # temporaries from some calls
        self._pnPointer = ct.byref(ct.c_int())
        self._puPointer = ct.byref(ct.c_uint())
        self._pbPointer = ct.byref(ct.c_bool())
        self._pdPointer = ct.byref(ct.c_double())

        self.CreateObject()

        # format for what this holds is: uModel, _pszSerialNum, uCOMPortID, uDeviceID, uChannelID
        self.mcDict = {}

        while True:
            if not len(self.mcDict):
                mcc = self.FindFirstMultiClamp()

            else:
                mcc = self.FindNextMultiClamp()

            if not mcc:
                break

            self.mcDict[self._generateUIDFromMCCTuple(mcc)] = mcc

        if not len(self.mcDict):
            raise ValueError("Could not find any MCC amplifiers!")

        # select the first one
        self.selectUniqueID(next(iter(self.mcDict)))

        if not self.CheckAPIVersion():
            raise IOError(f"The API version {API_VERSION_STR} is not supported")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def _handleError(self):
        """
        Return a tuple with the error state (T/F), the error value and the error message
        """

        errdict = {6000: 'MCCMSG_ERROR_NOERROR',
                   6001: 'MCCMSG_ERROR_OUTOFMEMORY',
                   6002: 'MCCMSG_ERROR_MCCNOTOPEN',
                   6003: 'MCCMSG_ERROR_INVALIDDLLHANDLE',
                   6004: 'MCCMSG_ERROR_INVALIDPARAMETER',
                   6005: 'MCCMSG_ERROR_MSGTIMEOUT',
                   6006: 'MCCMSG_ERROR_MCCCOMMANDFAIL'}

        errval = val(self._pnError, c_int_p)

        if errval != 6000:
            self._pnError = ct.byref(ct.c_int(6000))
            return True, errval, errdict[errval]

        return False, errval, ""

    def _getDLL(self, dllPath):
        olddir = os.getcwd()

        try:
            os.chdir(dllPath)
            self.aDLL = ct.windll.AxMultiClampMsg
            os.chdir(olddir)
        except IOError:
            os.chdir(olddir)
            raise IOError('Multiclamp DLL not found! Check your install path!')

    def _returnUID(self, serial, channel):
        """ give demo mccs a unique id channel correspondence"""
        if serial == 'Demo':  # we know we have at least one demo channel
            c1_count = 0
            c2_count = 0
            # have to deal with the fact that channels increment only AFTER the
            # first run through
            for uniqueID, tup in self.mcDict.items():
                if uniqueID.count('Demo'):
                    chan = tup[-1]
                    if chan == 1:
                        c1_count += 1
                    elif chan == 2:
                        c2_count += 1
            if channel == 1:
                demo_count = c1_count + 1
            elif channel == 2:
                demo_count = c2_count + 1
            return serial + '%s' % demo_count
        else:
            return serial

    def _generateUIDFromMCCTuple(self, mcTuple):
        serial = val(mcTuple[1], ct.c_char_p).decode('utf-8')
        channel = mcTuple[-1]
        serial = serial.strip('(').rstrip(')')
        serial = self._returnUID(serial, channel)
        mcid = '%s_%s' % (serial, channel)
        return mcid

    def getSerial(self):
        return self.currentUniqueID.split('_')[0]

    def getChannel(self):
        mcTuple = self.mcDict[self.currentUniqueID]
        return mcTuple[-1]

    def getUIDs(self):
        """
        Return a list of all amplifier UIDs
        """

        return self.mcDict.keys()

    def cleanup(self):
        self.DestroyObject()

    def selectUniqueID(self, uniqueID):
        try:
            mcTup = self.mcDict[uniqueID]
        except KeyError:
            print(self.mcDict.keys())
            raise KeyError(f"I dont know where you got the UID {uniqueID} but it wasn't from here! Check your config!")

        out = self.SelectMultiClamp(*mcTup)
        self.currentUniqueID = uniqueID
        return out

    """everything below interfaces with the MCC SDK API through ctypes"""

# DLL functions
    def CreateObject(self):
        """run this first to create self.hMCCmsg"""
        self.hMCCmsg = self.aDLL.MCCMSG_CreateObject(self._pnError)
        return self.hMCCmsg

    def DestroyObject(self):
        """Do this last if you do it"""
        return self.aDLL.MCCMSG_DestroyObject(self.hMCCmsg)

# General funcs
    def SetTimeOut(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_SetTimeOut(self.hMCCmsg, uValue, self._pnError)
        return self._handleError()

    def FindFirstMultiClamp(self):
        """
        Find the first available amplifier and return its parameter
        """

        puModel = ct.byref(ct.c_uint(0))
        pszSerialNum = ct.byref(ct.c_char_p(b''))
        uBufSize = ct.c_uint(16)
        puCOMPortID = ct.byref(ct.c_uint(0))
        puDeviceID = ct.byref(ct.c_uint(0))
        puChannelID = ct.byref(ct.c_uint(0))  # head stage

        if self.aDLL.MCCMSG_FindFirstMultiClamp(self.hMCCmsg, puModel, pszSerialNum,
                                                uBufSize, puCOMPortID, puDeviceID,
                                                puChannelID, self._pnError):
            return (val(puModel, c_uint_p), pszSerialNum,
                    val(puCOMPortID, c_uint_p),
                    val(puDeviceID, c_uint_p),
                    val(puChannelID, c_uint_p))

        return None

    def FindNextMultiClamp(self):
        """
        Find the next available amplifier and return its parameter
        """

        puModel = ct.byref(ct.c_uint(0))
        pszSerialNum = ct.byref(ct.c_char_p(b''))
        uBufSize = ct.c_uint(16)
        puCOMPortID = ct.byref(ct.c_uint(0))
        puDeviceID = ct.byref(ct.c_uint(0))
        puChannelID = ct.byref(ct.c_uint(0))  # head stage

        if self.aDLL.MCCMSG_FindNextMultiClamp(self.hMCCmsg, puModel, pszSerialNum,
                                               uBufSize, puCOMPortID, puDeviceID,
                                               puChannelID, self._pnError):
            return (val(puModel, c_uint_p), pszSerialNum,
                    val(puCOMPortID, c_uint_p),
                    val(puDeviceID, c_uint_p),
                    val(puChannelID, c_uint_p))

        return None

    def SelectMultiClamp(self, uModel, _pszSerialNum, uCOMPortID, uDeviceID, uChannelID):
        self.aDLL.MCCMSG_SelectMultiClamp(self.hMCCmsg, uModel, _pszSerialNum,
                                          uCOMPortID, uDeviceID, uChannelID, self._pnError)
        return self._handleError()

    def CheckAPIVersion(self):
        queryVersion = ct.c_char_p(API_VERSION_STR)
        self.aDLL.MCCMSG_CheckAPIVersion(queryVersion, self._pnError)
        return self._handleError()

    def BuildErrorText(self):
        errval = val(self._pnError, c_int_p)
        txtBufLen = 256
        txtBuf = ct.c_char_p(b' ' * txtBufLen)

        self.aDLL.MCCMSG_BuildErrorText(self.hMCCmsg, errval, txtBuf, txtBufLen, self._pnError)
        return self._handleError(), val(txtBuf, ct.c_char_p).decode('utf-8')

# MCC mode funcs
    def SetMode(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_SetMode(self.hMCCmsg, uValue, self._pnError)
        return self._handleError()

    def GetMode(self):
        self.aDLL.MCCMSG_GetMode(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetModeSwitchEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetModeSwitchEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetModeSwitchEnable(self):
        self.aDLL.MCCMSG_GetModeSwitchEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

# MCC holding funcs
    def SetHoldingEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetHoldingEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetHoldingEnable(self):
        self.aDLL.MCCMSG_GetHoldingEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetHolding(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetHolding(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetHolding(self):
        self.aDLL.MCCMSG_GetHolding(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC seal test and tuning funcs
    def SetTestSignalEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetTestSignalEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetTestSignalEnable(self):
        self.aDLL.MCCMSG_GetTestSignalEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetTestSignalAmplitude(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetTestSignalAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetTestSignalAmplitude(self):
        self.aDLL.MCCMSG_GetTestSignalAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetTestSignalFrequency(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetTestSignalFrequency(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetTestSignalFrequency(self):
        self.aDLL.MCCMSG_GetTestSignalFrequency(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC pipette offset funcs
    def AutoPipetteOffset(self):
        self.aDLL.MCCMSG_AutoPipetteOffset(self.hMCCmsg, self._pnError)
        return self._handleError()

    def SetPipetteOffset(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPipetteOffset(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPipetteOffset(self):
        self.aDLL.MCCMSG_GetPipetteOffset(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# IC ONLY MCC inject slow current
    def SetSlowCurrentInjEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetSlowCurrentInjEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetSlowCurrentInjEnable(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetSlowCurrentInjLevel(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSlowCurrentInjLevel(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSlowCurrentInjLevel(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjLevel(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCurrentInjSettlingTime(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSlowCurrentInjSettlingTime(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSlowCurrentInjSettlingTime(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjSettlingTime(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# VC ONLY MCC compensation funcs
    def SetFastCompCap(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetFastCompCap(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetFastCompCap(self):
        self.aDLL.MCCMSG_GetFastCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompCap(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSlowCompCap(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSlowCompCap(self):
        self.aDLL.MCCMSG_GetSlowCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetFastCompTau(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetFastCompTau(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetFastCompTau(self):
        self.aDLL.MCCMSG_GetFastCompTau(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompTau(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSlowCompTau(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSlowCompTau(self):
        self.aDLL.MCCMSG_GetSlowCompTau(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompTauX20Enable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetSlowCompTauX20Enable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetSlowCompTauX20Enable(self):
        self.aDLL.MCCMSG_GetSlowCompTauX20Enable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def AutoFastComp(self):
        self.aDLL.MCCMSG_AutoFastComp(self.hMCCmsg, self._pnError)
        return self._handleError()

    def AutoSlowComp(self):
        self.aDLL.MCCMSG_AutoSlowComp(self.hMCCmsg, self._pnError)
        return self._handleError()

# IC ONLY MCC pipette capacitance neutralization funcs
    def SetNeutralizationEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetNeutralizationEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetNeutralizationEnable(self):
        self.aDLL.MCCMSG_GetNeutralizationEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetNeutralizationCap(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetNeutralizationCap(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetNeutralizationCap(self):
        self.aDLL.MCCMSG_GetNeutralizationCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# VC ONLY MCC whole cell funcs
    def SetWholeCellCompEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetWholeCellCompEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetWholeCellCompEnable(self):
        self.aDLL.MCCMSG_GetWholeCellCompEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetWholeCellCompCap(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetWholeCellCompCap(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetWholeCellCompCap(self):
        self.aDLL.MCCMSG_GetWholeCellCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetWholeCellCompResist(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetWholeCellCompResist(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetWholeCellCompResist(self):
        self.aDLL.MCCMSG_GetWholeCellCompResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoWholeCellComp(self):
        self.aDLL.MCCMSG_AutoWholeCellComp(self.hMCCmsg, self._pnError)
        return self._handleError()

# VC ONLY MCC rs compensation funcs
    def SetRsCompEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetRsCompEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetRsCompEnable(self):
        self.aDLL.MCCMSG_GetRsCompEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetRsCompBandwidth(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetRsCompBandwidth(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetRsCompBandwidth(self):
        self.aDLL.MCCMSG_GetRsCompBandwidth(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetRsCompCorrection(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetRsCompCorrection(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetRsCompCorrection(self):
        self.aDLL.MCCMSG_GetRsCompCorrection(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetRsCompPrediction(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetRsCompPrediction(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetRsCompPrediction(self):
        self.aDLL.MCCMSG_GetRsCompPrediction(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC oscillation killer funcs
    def SetOscKillerEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetOscKillerEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetOscKillerEnable(self):
        self.aDLL.MCCMSG_GetOscKillerEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

# MCC primary (or scaled) signal funcs
    def SetPrimarySignal(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_SetPrimarySignal(self.hMCCmsg, uValue, self._pnError)
        return self._handleError()

    def GetPrimarySignal(self):
        self.aDLL.MCCMSG_GetPrimarySignal(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetPrimarySignalGain(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalGain(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPrimarySignalGain(self):
        self.aDLL.MCCMSG_GetPrimarySignalGain(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPrimarySignalLPF(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPrimarySignalLPF(self):
        self.aDLL.MCCMSG_GetPrimarySignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPrimarySignalHPF(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalHPF(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPrimarySignalHPF(self):
        self.aDLL.MCCMSG_GetPrimarySignalHPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC scope signal funcs
    def SetScopeSignalLPF(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetScopeSignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetScopeSignalLPF(self):
        self.aDLL.MCCMSG_GetScopeSignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC secondary (or raw) signal funcs
    def SetSecondarySignal(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_SetSecondarySignal(self.hMCCmsg, uValue, self._pnError)
        return self._handleError()

    def GetSecondarySignal(self):
        self.aDLL.MCCMSG_GetSecondarySignal(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetSecondarySignalGain(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSecondarySignalGain(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSecondarySignalGain(self):
        self.aDLL.MCCMSG_GetSecondarySignalGain(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSecondarySignalLPF(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetSecondarySignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetSecondarySignalLPF(self):
        self.aDLL.MCCMSG_GetSecondarySignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC output zero funcs
    def SetOutputZeroEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetOutputZeroEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetOutputZeroEnable(self):
        self.aDLL.MCCMSG_GetOutputZeroEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetOutputZeroAmplitude(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetOutputZeroAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetOutputZeroAmplitude(self):
        self.aDLL.MCCMSG_GetOutputZeroAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoOutputZero(self):
        self.aDLL.MCCMSG_AutoOutputZero(self.hMCCmsg, self._pnError)
        return self._handleError()

# VC ONLY MCC leak subtraction funcs
    def SetLeakSubEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetLeakSubEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetLeakSubEnable(self):
        self.aDLL.MCCMSG_GetLeakSubEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetLeakSubResist(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetLeakSubResist(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetLeakSubResist(self):
        self.aDLL.MCCMSG_GetLeakSubResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoLeakSub(self):
        self.aDLL.MCCMSG_AutoLeakSub(self.hMCCmsg, self._pnError)
        return self._handleError()

# IC ONLY MCC bridge balance funcs
    def SetBridgeBalEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetBridgeBalEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetBridgeBalEnable(self):
        self.aDLL.MCCMSG_GetBridgeBalEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetBridgeBalResist(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetBridgeBalResist(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetBridgeBalResist(self):
        self.aDLL.MCCMSG_GetBridgeBalResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoBridgeBal(self):
        self.aDLL.MCCMSG_AutoBridgeBal(self.hMCCmsg, self._pnError)
        return self._handleError()

# IC ONLY MCC clear funcs
    def ClearPlus(self):
        self.aDLL.MCCMSG_ClearPlus(self.hMCCmsg, self._pnError)
        return self._handleError()

    def ClearMinus(self):
        self.aDLL.MCCMSG_ClearMinus(self.hMCCmsg, self._pnError)
        return self._handleError()

# MCC pulse zap buzz!
    def Pulse(self):
        self.aDLL.MCCMSG_Pulse(self.hMCCmsg, self._pnError)
        return self._handleError()

    def SetPulseAmplitude(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPulseAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPulseAmplitude(self):
        self.aDLL.MCCMSG_GetPulseAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPulseDuration(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetPulseDuration(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetPulseDuration(self):
        self.aDLL.MCCMSG_GetPulseDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def Zap(self):
        self.aDLL.MCCMSG_Zap(self.hMCCmsg, self._pnError)
        return self._handleError()

    def SetZapDuration(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetZapDuration(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetZapDuration(self):
        self.aDLL.MCCMSG_GetZapDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def Buzz(self):
        self.aDLL.MCCMSG_Buzz(self.hMCCmsg, self._pnError)
        return self._handleError()

    def SetBuzzDuration(self, d):
        dValue = ct.c_double(d)
        self.aDLL.MCCMSG_SetBuzzDuration(self.hMCCmsg, dValue, self._pnError)
        return self._handleError()

    def GetBuzzDuration(self):
        self.aDLL.MCCMSG_GetBuzzDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC meter funcs
    def SetMeterResistEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetMeterResistEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetMeterResistEnable(self):
        self.aDLL.MCCMSG_GetMeterResistEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetMeterIrmsEnable(self, b):
        bValue = ct.c_bool(b)
        self.aDLL.MCCMSG_SetMeterIrmsEnable(self.hMCCmsg, bValue, self._pnError)
        return self._handleError()

    def GetMeterIrmsEnable(self):
        self.aDLL.MCCMSG_GetMeterIrmsEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def GetMeterValue(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_GetMeterValue(self.hMCCmsg, self._pdPointer, uValue, self._pnError)
        return val(self._pdPointer, c_double_p)

# MCC toolbar funcs
    def Reset(self):
        self.aDLL.MCCMSG_Reset(self.hMCCmsg, self._pnError)
        return self._handleError()

    def ToggleAlwaysOnTop(self):
        self.aDLL.MCCMSG_ToggleAlwaysOnTop(self.hMCCmsg, self._pnError)
        return self._handleError()

    def ToggleResize(self):
        self.aDLL.MCCMSG_ToggleResize(self.hMCCmsg, self._pnError)
        return self._handleError()

    def QuickSelectButton(self, u):
        uValue = ct.c_uint(u)
        self.aDLL.MCCMSG_QuickSelectButton(self.hMCCmsg, uValue, self._pnError)
        return self._handleError()


def main():

    with MultiClampControl() as mcc:
        outputFile = "mcc-output.json"
        with open(outputFile, mode="w", encoding="utf-8") as fh:
            d = DataGatherer()
            data = d.getData(mcc)
            data["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
            fh.write(json.dumps(data, sort_keys=True, indent=4))
            print(f"Output written to {outputFile}")

if __name__ == '__main__':
    main()

"""
primary signal reference
const UINT MCCMSG_PRI_SIGNAL_VC_MEMBCURRENT             = 0;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_VC_MEMBPOTENTIAL           = 1;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_VC_PIPPOTENTIAL            = 2;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_VC_100XACMEMBPOTENTIAL     = 3;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_VC_EXTCMDPOTENTIAL         = 4;  // 700B only
const UINT MCCMSG_PRI_SIGNAL_VC_AUXILIARY1              = 5;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_VC_AUXILIARY2              = 6;  // 700B only

const UINT MCCMSG_PRI_SIGNAL_IC_MEMBPOTENTIAL           = 7;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_IC_MEMBCURRENT             = 8;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_IC_CMDCURRENT              = 9;  // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_IC_100XACMEMBPOTENTIAL     = 10; // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_IC_EXTCMDCURRENT           = 11; // 700B only
const UINT MCCMSG_PRI_SIGNAL_IC_AUXILIARY1              = 12; // 700B and 700A
const UINT MCCMSG_PRI_SIGNAL_IC_AUXILIARY2              = 13; // 700B only

// Parameters for MCCMSG_GetMeterValue()
const UINT MCCMSG_METER1                                = 0;  // 700B
const UINT MCCMSG_METER2                                = 1;  // 700B
const UINT MCCMSG_METER3                                = 2;  // 700B
const UINT MCCMSG_METER4                                = 3;  // 700B
"""
