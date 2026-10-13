#!/usr/bin/env python3
"""Plot every emission spectrum and decay curve in a .gxz file.

    python examples/plot_overview.py pomiary.gxz [out.png]
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import felixgx

f = felixgx.load(sys.argv[1])
print(f.summary())

fig, (ax_em, ax_dec) = plt.subplots(1, 2, figsize=(13, 5))

for m in f.emissions:
    tr = m.best                      # corrected curve if present, else raw
    ax_em.plot(tr.x, tr.y / np.nanmax(tr.y), label=f"{m.name} (ex {m.excitation_wavelength:g} nm)")
ax_em.set(xlabel="emission wavelength [nm]", ylabel="normalised intensity", title="Emission scans")
ax_em.legend(fontsize=6)

for m in f.decays:
    ax_dec.semilogy(m.raw.x, m.raw.y, lw=0.8, label=f"{m.name} ({m.emission_wavelength:g} nm)")
ax_dec.set(xlabel="time [ns]", ylabel="counts", title="Decay histograms")
ax_dec.legend(fontsize=6)

fig.tight_layout()
if len(sys.argv) > 2:
    fig.savefig(sys.argv[2], dpi=110)
else:
    plt.show()
