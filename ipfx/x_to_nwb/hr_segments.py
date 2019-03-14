"""
Create a numpy array with the stimulus set data from the stored parameters.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import scipy.signal


def getSegmentClass(stimRec, channelRec, segmentRec):
    """
    Return the correct derived class instance of Segment for the given records.
    """

    if segmentRec.Class == "Squarewave":
        return SquareSegment(stimRec, channelRec, segmentRec)
    elif segmentRec.Class == "Constant":
        return ConstantSegment(stimRec, channelRec, segmentRec)
    elif segmentRec.Class == "Ramp":
        return RampSegment(stimRec, channelRec, segmentRec)
    elif segmentRec.Class == "Chirpwave":
        return ChirpSegment(stimRec, channelRec, segmentRec)
    else:
        raise ValueError(f"Unsupported stim segment class {segmentRec.Class}")


# Use Replay->Show PGF Template in PatchMaster to view the stimset of the current trace
#
# PatchMaster manual page 113
# StimulationRecord.SampleInterval: Determines x-spacing
# StimSegmentRecord.Duration [s]: x-Length
# Amplitude [V] for VC and [µA] for IC
class Segment(ABC):
    """
        Base class for all segment types.
        Derived class must implement `createArray` only.

        The following segment types are supported:
        - Constant
        - Ramp
        - Square
        - Chirp

        Support for the following types is missing:
        - Continous
        - Sine

        Note: Only currently used segment options/modes/specialities are implemented.
    """

    def __init__(self, stimRec, channelRec, segmentRec):
        self.xDelta = {"mode": segmentRec.DurationIncMode,
                       "factor": segmentRec.DeltaTFactor,
                       "inc": segmentRec.DeltaTIncrement}

        self.yDelta = {"mode": segmentRec.VoltageIncMode,
                       "factor": segmentRec.DeltaVFactor,
                       "inc": segmentRec.DeltaVIncrement}

        self.amplitude = self.getAmplitude(channelRec, segmentRec)
        self.duration = segmentRec.Duration
        self.sampleInterval = stimRec.SampleInterval

        # PatchMaster stores "V" for VC and "µA" for IC
        # and we want to have "mV" for VC and "pA" for IC
        self.amplitudeScale = 1e3

        if not channelRec.StimToDacID['UseStimScale']:
            raise ValueError("The flag UseStimScale of StimToDacID being false is not supported.")

    def __str__(self):
        return ("xDelta={}, yDelta={}, "
                "duration={}, sampleInterval={}"
                "amplitudeScale={}").format(self.xDelta, self.yDelta,
                                            self.duration, self.sampleInterval,
                                            self.amplitudeScale)

    @staticmethod
    def _hasDelta(deltaDict):
        """
        Return true if delta mode is active, false otherwise.
        """

        return deltaDict["factor"] != 1.0 or deltaDict["inc"] != 0.0

    @staticmethod
    def _applyDelta(deltaDict, val, sweepNo):
        """
        Apply delta mode properties to val if active.

        Return the possibly modified value.
        """

        if not Segment._hasDelta(deltaDict):
            return val

        if deltaDict["mode"] != "LogInc" and deltaDict["mode"] != "Inc":
            raise ValueError(f"Increment mode {deltaDict['mode']} is not supported.")

        if deltaDict["factor"] == 1.0:
            return val + deltaDict["inc"] * sweepNo
        elif deltaDict["mode"] == "LogInc":
            return val + deltaDict["inc"] * math.exp(math.log(deltaDict["factor"]) * (sweepNo - 1))
        else:
            return val + deltaDict["inc"] * sweepNo + val * (math.exp(math.log(deltaDict["factor"]) * sweepNo) - 1.0)

    def applyAmplitudeScale(self, amplitude):
        """
        Scale the amplitude so that we get the correct units. See also Segment.createArray.
        """

        return amplitude * self.amplitudeScale

    def _step(self, xValue, yValue, sweepNo):
        """
        Apply delta modes to the given x and y values.

        Return the possibly modified values.
        """

        return Segment._applyDelta(self.xDelta, xValue, sweepNo), Segment._applyDelta(self.yDelta, yValue, sweepNo)

    @abstractmethod
    def createArray(self, sweep):
        """
        Return a numpy array with the stimset data.

        Units are [mV] for voltage clamp and [pA] for current clamp.
        """

        pass

    def hasXDelta(self):
        """
        Return true if delta mode is active for the x dimension.
        """

        return Segment._hasDelta(self.xDelta)

    def hasYDelta(self):
        """
        Return true if delta mode is active for the y dimension.
        """

        return Segment._hasDelta(self.yDelta)

    def doStepping(self, sweepNo):
        """
        Apply the delta modes the given number of times (once per sweep)
        """

        duration, amplitude = self._step(self.duration, self.amplitude, sweepNo)

        return duration, self.applyAmplitudeScale(amplitude)

    def calculateNumberOfPoints(self, duration):
        """
        Return the number of points of this segment.
        """
        return math.trunc(duration / self.sampleInterval)

    def getAmplitude(self, channelRec, segmentRec):
        """
        Extract the value of the amplitude from the ChannelRecordStimulus and the StimSegmentRecord.
        """

        if segmentRec.VoltageSource == "Constant":

            if channelRec.StimToDacID['UseRelative']:
                return segmentRec.Voltage + channelRec.Holding
            else:
                return segmentRec.Voltage

        elif segmentRec.VoltageSource == "Hold":
            return channelRec.Holding
        else:
            raise ValueError(f"Unsupported voltage source {segmentRec.VoltageSource}")


# Square Wave dialog in patchmaster:
# Peak Ampl. [V] -> ChannelRecordStimulus.Square_PosAmpl
# Neg. Ampl. [V] -> ChannelRecordStimulus.Square_NegAmpl
# Requested/Actual Frequency [Hz] -> ChannelRecordStimulus.Square_Cycle: duration [s]
# Pos. Dur. Factor -> ChannelRecordStimulus.Square_DurFactor, value of zero means no negative amplitude
# Base Incr. [mV] -> ChannelRecordStimulus.Square_BaseIncr [V]
# Top info box: Square Kind
class SquareSegment(Segment):

    def __init__(self, stimRec, channelRec, segmentRec):
        super().__init__(stimRec, channelRec, segmentRec)

        self.posAmp = channelRec.Square_PosAmpl
        self.negAmp = channelRec.Square_NegAmpl
        self.cycleDuration = channelRec.Square_Cycle
        self.durationFactor = channelRec.Square_DurFactor
        self.baseIncr = channelRec.Square_BaseIncr
        self.kind = channelRec.Square_Kind

        if self.baseIncr != 0:
            raise ValueError(f"Unsupported baseIncr={self.baseIncr}")
        elif self.durationFactor != 0:
            raise ValueError(f"Unsupported durationFactor={self.durationFactor}")
        elif self.kind != "Common Frequency":
            raise ValueError(f"Unsupported squareKind={self.squareKind}")
        elif self.hasXDelta() or self.hasYDelta():
            raise ValueError(f"Delta modes are not supported.")
        elif not (self.cycleDuration > 0):
            raise ValueError(f"Invalid cycle duration.")

    def __str__(self):
        return super().__str__() + \
               (", "
                "+amp={}, -amp={}, "
                "cycleDur={}, durFactor={}, "
                "baseIncr={}, squareKind={}").format(self.posAmp, self.negAmp,
                                                     self.cycleDuration, self.durationFactor,
                                                     self.baseIncr, self.kind)

    def getAmplitude(self, channelRec, segmentRec):
        return None

    def createArray(self, sweep):

        numPoints = self.calculateNumberOfPoints(self.duration)
        numPointsCycle = math.trunc(self.cycleDuration / self.sampleInterval)
        oneCycle = np.zeros([numPointsCycle])

        halfCycleLength = int(numPointsCycle/2)
        oneCycle[:halfCycleLength] = self.applyAmplitudeScale(self.posAmp)
        oneCycle[halfCycleLength:] = self.applyAmplitudeScale(-self.posAmp)

        numCycles = math.ceil(self.duration / self.cycleDuration)

        segment = np.tile(oneCycle, numCycles)
        segment.resize(numPoints)

        return segment


class ConstantSegment(Segment):

    def __init__(self, stimRec, channelRec, segmentRec):
        super().__init__(stimRec, channelRec, segmentRec)

    def __str__(self):
        return super().__str__() + \
               (", "
                "amp={}").format(self.amplitude)

    def createArray(self, sweep):

        duration, amplitude = self.doStepping(sweep)
        numPoints = self.calculateNumberOfPoints(duration)

        return np.full((numPoints), amplitude)


class RampSegment(Segment):

    def __init__(self, stimRec, channelRec, segmentRec):
        super().__init__(stimRec, channelRec, segmentRec)

    def __str__(self):
        return super().__str__() + \
               (", "
                "amp={}").format(self.amplitude)

    def createArray(self, sweep):

        duration, amplitude = self.doStepping(sweep)
        numPoints = self.calculateNumberOfPoints(duration)

        return np.linspace(0.0, amplitude, numPoints)


# Chirp wave dialog in PatchMaster:
# Top info box -> ChannelRecordStimulus.Chirp_Kind
# Amplitude -> ChannelRecordStimulus.Chirp_Amplitude, half of the peak to peak amplitude
# Start Frequency -> ChannelRecordStimulus.Chirp_StartFreq
# End Frequency -> ChannelRecordStimulus.Chirp_EndFreq
# Min. Points / Cycle -> ChannelRecordStimulus.Chirp_MinPoints (calculated)
# Segment Points is calculated
class ChirpSegment(Segment):

    def __init__(self, stimRec, channelRec, segmentRec):
        super().__init__(stimRec, channelRec, segmentRec)

        self.startFreq = channelRec.Chirp_StartFreq
        self.endFreq = channelRec.Chirp_EndFreq
        self.kind = channelRec.Chirp_Kind

        if self.kind != "Exponential" and self.kind != "Linear":
            raise ValueError(f"The chirp kind {self.kind} is not supported.")

    def __str__(self):
        return super().__str__() + \
                (", "
                 "amp {}, start freq {}, end freq = {}, chirp kind = {}"
                 ).format(self.amplitude, self.startFreq, self.endFreq, self.kind)

    def getAmplitude(self, channelRec, segmentRec):
        # The amplitude is half of the peak-to-peak amplitude and that is
        # stored in the DAT file.
        return channelRec.Chirp_Amplitude * 0.5

    def createArray(self, sweep):

        duration, amplitude = self.doStepping(sweep)
        numPoints = self.calculateNumberOfPoints(duration)
        x = np.linspace(0, duration, numPoints)

        if self.kind == "Exponential":
            method = "logarithmic"
        elif self.kind == "Linear":
            method = "linear"

        return amplitude * scipy.signal.chirp(x, f0=self.startFreq, f1=self.endFreq,
                                              t1=duration, method=method, phi=-90)
