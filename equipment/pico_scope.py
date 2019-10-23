import time

import numpy as np
from picoscope import ps2000

"""
Uses https://github.com/colinoflynn/pico-python
pip install picoscope
"""


class Scope:

    def __init__(self):
        self.ps = ps2000.PS2000()
        print(self.ps.getAllUnitInfo())
        waveform_desired_duration = 200E-3
        obs_duration = 3 * waveform_desired_duration
        sampling_interval = obs_duration / 4096
        (self.actualSamplingInterval, self.nSamples, maxSamples) = \
            self.ps.setSamplingInterval(sampling_interval, obs_duration)
        print('actual sampling interval = ', self.actualSamplingInterval)
        print('nsamples = ', self.nSamples)
        self.ps.setChannel('A', 'AC', 2.0, 0.0, enabled=True,
                                     BWLimited=False)
        self.ps.setSimpleTrigger('A', 0, 'Falling', timeout_ms=100,
                                 enabled=True)
        self.ps.setChannel('B', 'AC', 2.0, 0.0, enabled=True, BWLimited=False)
        self.ps.setSimpleTrigger('B', 0, 'Falling', timeout_ms=100,
                                 enabled=True)

    def get_V(self, refine_range=False, channel='A'):
        s = time.time()
        if refine_range:
            channelRange = self.ps.setChannel(channel, 'AC', 2.0, 0.0,
                                              enabled=True, BWLimited=False)
            self.ps.runBlock()
            self.ps.waitReady()
            data = self.ps.getDataV(channel, self.nSamples,
                                    returnOverflow=False)
            vrange = np.max(data) * 1.5
            channelRange = self.ps.setChannel(channel, 'AC', vrange, 0.0,
                                              enabled=True, BWLimited=False)
        self.ps.runBlock()
        self.ps.waitReady()
        data = self.ps.getDataV(channel, self.nSamples, returnOverflow=False)
        times = np.arange(self.nSamples)*self.actualSamplingInterval
        return times, data, time.time() - s






if __name__=="__main__":
    S = Scope()
    t, V, l = S.get_V(refine_range=True)
    print(l)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(t, V, 'x')
    plt.show()
    t, V, l = S.get_V(refine_range=True)
    plt.figure()
    plt.plot(t, V)
    plt.show()
    # S = Scope()