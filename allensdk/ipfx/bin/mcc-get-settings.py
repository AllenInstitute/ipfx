from ctypes import *
from time import sleep
#import inspect as ins
import os

#developed for api version 1.0.0.7

"""HOW TO USE:

    YOU MUST CALL cleanup() ON YOUR OWN

    if there are no erros then the object has dected all the mcs and is set on the first
    use selectMC(num) num [0:n-1] where n=num multiclamps
use FindFirst and FindNext to find the multiclamps,
then use the return values from those to slect which multiclamp you want to use
once you have selected a multiclamp all the get functions may be call directly without any args
auto functions will also be called without args
set functions will take only the value of the variable you want to set

basically, switch between headstages/multiclamps and then call the functions you want
"""



#mode defintion
MCC_MODE_DICT={0:'VC',1:'IC',2:'IEZ'}

#ptypes
c_int_p=POINTER(c_int)
c_uint_p=POINTER(c_uint)
c_bool_p=POINTER(c_bool)
c_double_p=POINTER(c_double)
c_string_p=POINTER(c_char)

#utility function for type casting and returning the value at a pointer
def val(ptr, ptype):
    if ptype==c_char_p:
        return cast(ptr,ptype).value
    return cast(ptr,ptype)[0]

#error handling
errdict={6000:'MCCMSG_ERROR_NOERROR', 6001:'MCCMSG_ERROR_OUTOFMEMORY',\
         6002:'MCCMSG_ERROR_MCCNOTOPEN', 6003:'MCCMSG_ERROR_INVALIDDLLHANDLE',\
         6004:'MCCMSG_ERROR_INVALIDPARAMETER', 6005:'MCCMSG_ERROR_MSGTIMEOUT',\
         6006:'MCCMSG_ERROR_MCCCOMMANDFAIL'}

class mccControl:
    def __init__(self,dllPath=None): #use this one for now
        #print(dllPath)
        self.mccDllPath='C:/Axon/MultiClamp 700B Commander/3rd Party Support/AxMultiClampMsg/'
        if dllPath:
            self.mccDllPath=dllPath
        self.getDLL() #FIXME this fails silently on 64bit see if we can make it more explicit
        self._pnError=byref(c_int()) #err pointer
        self.CreateObject() #create the dll handle NOTE: this MUST be called EVERY time
        #check for the first MC, if we don't find it threaded sleep do it again
        #once we find the MCs then SAVE THE POINTERS to the serial numbers and give THOSE to the new handler every time
        self._puModel=byref(c_uint())
        self._pszSerialNum=byref(c_char_p(b''))
        self.uBufSize=c_uint(16) #just setting this manually, shouldn't be anything other than 16
        self._puCOMPortID=byref(c_uint())
        self._puDeviceID=byref(c_uint())
        self._puChannelID=byref(c_uint()) #head stage, need a way to switch this quickly
        try:
            firstMC=self.FindFirstMultiClamp()
        except:
            raise IOError('no multiclamps found, is mcc on?')
        self.getMCS(firstMC)
        self.selectMC(0)
        self._pnPointer=byref(c_int())
        self._puPointer=byref(c_uint())
        self._pbPointer=byref(c_bool())
        self._pdPointer=byref(c_double())

    def __enter__(self):
        return self

    def __exit__(self,type,value,traceback):
        self.DestroyObject()

    def errPrint(self):
        errval=val(self._pnError,c_int_p)
        if errval==6000:
            return errval
        else:
            self._pnError=byref(c_int(6000))
            printD(errdict[errval])
            return errval

    def getDLL(self):
        try:
            olddir=os.getcwd()
            os.chdir(self.mccDllPath)
            #print(self.mccDllPath)
            #print(os.getcwd())
            self.aDLL=windll.AxMultiClampMsg #magic!
            os.chdir(olddir)
            #print(olddir)
        except: #I need to catch the 32bit 64bit conflict here
            print('Multiclamp DLL not found! Check your install path!')
            raise

    def demoCheck(self,serial,channel):
        """ give demo mccs a unique id channle corrispondence will be wrong? """
        if serial == 'Demo': #we know we have at least one demo channel
            c1_count = 0
            c2_count = 0
            for uniqueID,tup in self.mcDict.items(): #have to deal with the fact that channels increment only AFTER the first run through
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
            return serial+'%s'%demo_count
        else:
            return serial

    def uniqueID(self,mcTuple): #FIXME this needs to be priv?
        serial=val(mcTuple[1],c_char_p).decode('utf-8')
        channel=mcTuple[-1] #FIXME this tuple should really have a corrisponding class...
        serial=serial.strip('(').rstrip(')')
        serial = self.demoCheck(serial,channel)
        mcid='%s_%s'%(serial,channel) #UID DEFINITION RIGHTHERE XXX
        return mcid

    def getSerial(self):
        #mcTuple=self.mcDict[self.currentUniqueID]
        #serial=val(mcTuple[1],c_char_p).decode('utf-8')
        #serial=serial.strip('(').rstrip(')')
        #serial = self.demoCheck(serial) #FIXME this is not deterministic
        return self.currentUniqueID.split('_')[0] #well ok...

    def getChannel(self):
        mcTuple=self.mcDict[self.currentUniqueID]
        return mcTuple[-1] #FIXME this tuple should really have a corrisponding class...

    def getMCS(self,firstMC):
        """get all the multiclamps and store them in a list"""
        #FIXME NO the problem is NOT with lossing the pointer to the SN, any pointer will do
        if type(firstMC)==tuple:
            #format for what this holds is: uModel, _pszSerialNum, uCOMPortID, uDeviceID, uChannelID
            self.mcList=[]
            self.mcDict={}
            self.mcList.append(firstMC)
            self.mcDict[self.uniqueID(firstMC)]=firstMC
            #printD(firstMC,val(firstMC[1],c_char_p))
            while 1:
                nextMC=self.FindNextMultiClamp()
                if nextMC:
                    self.mcList.append(nextMC)
                    self.mcDict[self.uniqueID(nextMC)]=nextMC
                else:
                    #print(self.mcNum,"multiclamps found!")
                    self.mcNum=len(self.mcList)
                    break
        else:
            print('No multiclamps found! MCC probably isnt on! Crashing!')

    def cleanup(self):
        """called in __exit__ to make sure we have no memory leeks"""
        self.DestroyObject()
        print('hMCCmsg successfully removed, no memory leaks here!')

    def selectMC(self,num): #XXX deprecated
        return None
        try:
            if num <= (self.mcNum):
                out = self.SelectMultiClamp(*self.mcList[num]) #FIXME errors be here
                self.mcCurrent=num
                self.currentUniqueID=self.uniqueID(self.mcList[num]) #FIXME mcDict?
                return out
            else:
                print("You don't have that many multiclamps!")
                return 0
        except AttributeError:
            return None

    def selectUniqueID(self,uniqueID):
        try:
            mcTup=self.mcDict[uniqueID]
        except KeyError:
            print(self.mcDict.keys())
            raise KeyError('I dont know where you got uid "%s" but it wasnt from here! Check your config!'%uniqueID)
       
        self.mcCurrent=None #FIXME get rid of all this list nonsense
        out = self.SelectMultiClamp(*mcTup)
        self.currentUniqueID=uniqueID
        return out

    def selectNextMC(self): #XXX deprecated
        return None
        """Sets the currentMC to the next available in a loop"""
        num=(self.mcCurrent+1)%(self.mcNum)
        self.mcCurrent=num
        self.currentUniqueID=self.uniqueID(self.mcList[num]) #FIXME mcDict?
        return self.SelectMultiClamp(*self.mcList[self.currentMC])

    """everything below interfaces with the MCC SDK API through ctypes"""

    #def CheckAPIVersion(self): #FIXME
        #self.aDLL.MCCMSG_CheckAPIVersion(LPCSTR pszQueryVersion)

#DLL functions
    def CreateObject(self):
        """run this first to create self.hMCCmsg"""
        self.hMCCmsg=self.aDLL.MCCMSG_CreateObject(self._pnError)
        return self.hMCCmsg

    def DestroyObject(self):
        """Do this last if you do it and try to reinit by hand you are silly"""
        return self.aDLL.MCCMSG_DestroyObject(self.hMCCmsg)

#General funcs
    def SetTimeOut(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_SetTimeOut(self.hMCCmsg, uValue, self._pnError)
        return self.errPrint()

#MCC selection funcs
    def FindFirstMultiClamp(self):
        #the if statement is where most of the CPU is used, so not much we can do about that except sleep ;_;
        if self.aDLL.MCCMSG_FindFirstMultiClamp(self.hMCCmsg, self._puModel, self._pszSerialNum, self.uBufSize, self._puCOMPortID, self._puDeviceID, self._puChannelID, self._pnError):
            outTup=(val(self._puModel,c_uint_p),\
                    self._pszSerialNum,\
                    val(self._puCOMPortID,c_uint_p),\
                    val(self._puDeviceID,c_uint_p),\
                    val(self._puChannelID,c_uint_p))
            return outTup
        else:
            raise IOError

    def FindNextMultiClamp(self):
        _puModel=byref(c_uint(0))
        _pszSerialNum=byref(c_char_p(b''))
        uBufSize=c_uint(16) #just setting this manually, shouldn't be anything other than 16
        _puCOMPortID=byref(c_uint(0))
        _puDeviceID=byref(c_uint(0))
        _puChannelID=byref(c_uint(0)) #head stage, need a way to switch this quickly
        if self.aDLL.MCCMSG_FindNextMultiClamp(self.hMCCmsg, _puModel, _pszSerialNum, uBufSize, _puCOMPortID, _puDeviceID, _puChannelID, self._pnError):
            outTup=(
                    val(_puModel,c_uint_p),\
                    _pszSerialNum,\
                    val(_puCOMPortID,c_uint_p),\
                    val(_puDeviceID,c_uint_p),\
                    val(_puChannelID,c_uint_p)
            )
            return outTup
        else:
            return 0

    def SelectMultiClamp(self, uModel, _pszSerialNum, uCOMPortID, uDeviceID, uChannelID):
        self.aDLL.MCCMSG_SelectMultiClamp(self.hMCCmsg, uModel, _pszSerialNum, uCOMPortID, uDeviceID, uChannelID, self._pnError)
        return self.errPrint()

#MCC mode funcs
    def SetMode(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_SetMode(self.hMCCmsg, uValue, self._pnError)
        return self.errPrint()

    def GetMode(self):
        self.aDLL.MCCMSG_GetMode(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetModeSwitchEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetModeSwitchEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetModeSwitchEnable(self):
        self.aDLL.MCCMSG_GetModeSwitchEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

#MCC holding funcs
    def SetHoldingEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetHoldingEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetHoldingEnable(self):
        self.aDLL.MCCMSG_GetHoldingEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetHolding(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetHolding(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetHolding(self):
        self.aDLL.MCCMSG_GetHolding(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC seal test and tuning funcs
    def SetTestSignalEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetTestSignalEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetTestSignalEnable(self):
        self.aDLL.MCCMSG_GetTestSignalEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetTestSignalAmplitude(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetTestSignalAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetTestSignalAmplitude(self):
        self.aDLL.MCCMSG_GetTestSignalAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetTestSignalFrequency(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetTestSignalFrequency(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetTestSignalFrequency(self):
        self.aDLL.MCCMSG_GetTestSignalFrequency(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC pipette offset funcs
    def AutoPipetteOffset(self):
        self.aDLL.MCCMSG_AutoPipetteOffset(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def SetPipetteOffset(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPipetteOffset(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPipetteOffset(self):
        self.aDLL.MCCMSG_GetPipetteOffset(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#IC ONLY MCC inject slow current
    def SetSlowCurrentInjEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetSlowCurrentInjEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetSlowCurrentInjEnable(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetSlowCurrentInjLevel(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSlowCurrentInjLevel(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSlowCurrentInjLevel(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjLevel(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCurrentInjSettlingTime(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSlowCurrentInjSettlingTime(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSlowCurrentInjSettlingTime(self):
        self.aDLL.MCCMSG_GetSlowCurrentInjSettlingTime(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#VC ONLY MCC compensation funcs
    def SetFastCompCap(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetFastCompCap(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetFastCompCap(self):
        self.aDLL.MCCMSG_GetFastCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompCap(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSlowCompCap(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSlowCompCap(self):
        self.aDLL.MCCMSG_GetSlowCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetFastCompTau(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetFastCompTau(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetFastCompTau(self):
        self.aDLL.MCCMSG_GetFastCompTau(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompTau(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSlowCompTau(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSlowCompTau(self):
        self.aDLL.MCCMSG_GetSlowCompTau(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSlowCompTauX20Enable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetSlowCompTauX20Enable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetSlowCompTauX20Enable(self):
        self.aDLL.MCCMSG_GetSlowCompTauX20Enable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def AutoFastComp(self):
        self.aDLL.MCCMSG_AutoFastComp(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def AutoSlowComp(self):
        self.aDLL.MCCMSG_AutoSlowComp(self.hMCCmsg, self._pnError)
        return self.errPrint()

#IC ONLY MCC pipette capacitance neutralization funcs
    def SetNeutralizationEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetNeutralizationEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetNeutralizationEnable(self):
        self.aDLL.MCCMSG_GetNeutralizationEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetNeutralizationCap(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetNeutralizationCap(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetNeutralizationCap(self):
        self.aDLL.MCCMSG_GetNeutralizationCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#VC ONLY MCC whole cell funcs
    def SetWholeCellCompEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetWholeCellCompEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetWholeCellCompEnable(self):
        self.aDLL.MCCMSG_GetWholeCellCompEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetWholeCellCompCap(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetWholeCellCompCap(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetWholeCellCompCap(self):
        self.aDLL.MCCMSG_GetWholeCellCompCap(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetWholeCellCompResist(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetWholeCellCompResist(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetWholeCellCompResist(self):
        self.aDLL.MCCMSG_GetWholeCellCompResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoWholeCellComp(self):
        self.aDLL.MCCMSG_AutoWholeCellComp(self.hMCCmsg, self._pnError)
        return self.errPrint()

#VC ONLY MCC rs compensation funcs
    def SetRsCompEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetRsCompEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetRsCompEnable(self):
        self.aDLL.MCCMSG_GetRsCompEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetRsCompBandwidth(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetRsCompBandwidth(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetRsCompBandwidth(self):
        self.aDLL.MCCMSG_GetRsCompBandwidth(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetRsCompCorrection(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetRsCompCorrection(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetRsCompCorrection(self):
        self.aDLL.MCCMSG_GetRsCompCorrection(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetRsCompPrediction(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetRsCompPrediction(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetRsCompPrediction(self):
        self.aDLL.MCCMSG_GetRsCompPrediction(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC oscillation killer funcs
    def SetOscKillerEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetOscKillerEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetOscKillerEnable(self):
        self.aDLL.MCCMSG_GetOscKillerEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

#MCC primary (or scaled) signal funcs
    def SetPrimarySignal(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_SetPrimarySignal(self.hMCCmsg, uValue, self._pnError)
        return self.errPrint()

    def GetPrimarySignal(self):
        self.aDLL.MCCMSG_GetPrimarySignal(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetPrimarySignalGain(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalGain(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPrimarySignalGain(self):
        self.aDLL.MCCMSG_GetPrimarySignalGain(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPrimarySignalLPF(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPrimarySignalLPF(self):
        self.aDLL.MCCMSG_GetPrimarySignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPrimarySignalHPF(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPrimarySignalHPF(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPrimarySignalHPF(self):
        self.aDLL.MCCMSG_GetPrimarySignalHPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC scope signal funcs
    def SetScopeSignalLPF(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetScopeSignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetScopeSignalLPF(self):
        self.aDLL.MCCMSG_GetScopeSignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC secondary (or raw) signal funcs
    def SetSecondarySignal(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_SetSecondarySignal(self.hMCCmsg, uValue, self._pnError)
        return self.errPrint()

    def GetSecondarySignal(self):
        self.aDLL.MCCMSG_GetSecondarySignal(self.hMCCmsg, self._puPointer, self._pnError)
        return val(self._puPointer, c_uint_p)

    def SetSecondarySignalGain(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSecondarySignalGain(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSecondarySignalGain(self):
        self.aDLL.MCCMSG_GetSecondarySignalGain(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetSecondarySignalLPF(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetSecondarySignalLPF(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetSecondarySignalLPF(self):
        self.aDLL.MCCMSG_GetSecondarySignalLPF(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC output zero funcs
    def SetOutputZeroEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetOutputZeroEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetOutputZeroEnable(self):
        self.aDLL.MCCMSG_GetOutputZeroEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetOutputZeroAmplitude(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetOutputZeroAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetOutputZeroAmplitude(self):
        self.aDLL.MCCMSG_GetOutputZeroAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoOutputZero(self):
        self.aDLL.MCCMSG_AutoOutputZero(self.hMCCmsg, self._pnError)
        return self.errPrint()

#VC ONLY MCC leak subtraction funcs
    def SetLeakSubEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetLeakSubEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetLeakSubEnable(self):
        self.aDLL.MCCMSG_GetLeakSubEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetLeakSubResist(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetLeakSubResist(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetLeakSubResist(self):
        self.aDLL.MCCMSG_GetLeakSubResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoLeakSub(self):
        self.aDLL.MCCMSG_AutoLeakSub(self.hMCCmsg, self._pnError)
        return self.errPrint()

#IC ONLY MCC bridge balance funcs
    def SetBridgeBalEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetBridgeBalEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetBridgeBalEnable(self):
        self.aDLL.MCCMSG_GetBridgeBalEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetBridgeBalResist(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetBridgeBalResist(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetBridgeBalResist(self):
        self.aDLL.MCCMSG_GetBridgeBalResist(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def AutoBridgeBal(self):
        self.aDLL.MCCMSG_AutoBridgeBal(self.hMCCmsg, self._pnError)
        return self.errPrint()

#IC ONLY MCC clear funcs
    def ClearPlus(self):
        self.aDLL.MCCMSG_ClearPlus(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def ClearMinus(self):
        self.aDLL.MCCMSG_ClearMinus(self.hMCCmsg, self._pnError)
        return self.errPrint()

#MCC pulse zap buzz!
    def Pulse(self):
        self.aDLL.MCCMSG_Pulse(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def SetPulseAmplitude(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPulseAmplitude(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPulseAmplitude(self):
        self.aDLL.MCCMSG_GetPulseAmplitude(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def SetPulseDuration(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetPulseDuration(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetPulseDuration(self):
        self.aDLL.MCCMSG_GetPulseDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def Zap(self):
        self.aDLL.MCCMSG_Zap(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def SetZapDuration(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetZapDuration(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetZapDuration(self):
        self.aDLL.MCCMSG_GetZapDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

    def Buzz(self):
        self.aDLL.MCCMSG_Buzz(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def SetBuzzDuration(self, d):
        dValue=c_double(d)
        self.aDLL.MCCMSG_SetBuzzDuration(self.hMCCmsg, dValue, self._pnError)
        return self.errPrint()

    def GetBuzzDuration(self):
        self.aDLL.MCCMSG_GetBuzzDuration(self.hMCCmsg, self._pdPointer, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC meter funcs
    def SetMeterResistEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetMeterResistEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetMeterResistEnable(self):
        self.aDLL.MCCMSG_GetMeterResistEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def SetMeterIrmsEnable(self, b):
        bValue=c_bool(b)
        self.aDLL.MCCMSG_SetMeterIrmsEnable(self.hMCCmsg, bValue, self._pnError)
        return self.errPrint()

    def GetMeterIrmsEnable(self):
        self.aDLL.MCCMSG_GetMeterIrmsEnable(self.hMCCmsg, self._pbPointer, self._pnError)
        return val(self._pbPointer, c_bool_p)

    def GetMeterValue(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_GetMeterValue(self.hMCCmsg, self._pdPointer, uValue, self._pnError)
        return val(self._pdPointer, c_double_p)

#MCC toolbar funcs
    def Reset(self):
        self.aDLL.MCCMSG_Reset(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def ToggleAlwaysOnTop(self):
        self.aDLL.MCCMSG_ToggleAlwaysOnTop(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def ToggleResize(self):
        self.aDLL.MCCMSG_ToggleResize(self.hMCCmsg, self._pnError)
        return self.errPrint()

    def QuickSelectButton(self, u):
        uValue=c_uint(u)
        self.aDLL.MCCMSG_QuickSelectButton(self.hMCCmsg, uValue, self._pnError)
        return self.errPrint()

    #def BuildErrorText(self, s): #FIXME
        #self.aDLL.MCCMSG_BuildErrorText(self.hMCCmsg, nValue, sValue, uValue)
        #return self.errPrint()

def main():
    MCC_DLLPATH = 'C:/Axon/MultiClamp 700B Commander/3rd Party Support/AxMultiClampMsg/'
    with mccControl(MCC_DLLPATH) as mcc:
        for uid in mcc.mcDict:
            mcc.selectUniqueID(uid)
            vals = [mcc.GetMeterValue(i) for i in range(4)]
            print(vals)


if __name__=='__main__':
    main()


""" primary signal reference
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
