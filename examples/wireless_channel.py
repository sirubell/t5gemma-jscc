"""Replace the forward body with your Python wireless simulation.

This example is a deterministic attenuation channel, not a physical simulator.
"""
from torch import nn


class WirelessChannel(nn.Module):
    def __init__(self, gain=0.9):
        super().__init__()
        self.gain = gain

    def forward(self, z, snr_db):
        # z already has unit average power per sample. Preserve its shape,
        # device and dtype when returning the received real-valued symbols.
        return z if snr_db is None else z * self.gain
