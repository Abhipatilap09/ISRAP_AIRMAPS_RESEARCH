"""Training-only station/hour median used in the PM2.5 comparisons."""
import numpy as np
import pandas as pd
from .common import SITES

def hour_median(pm, times, rows):
    hour = (times - pd.Timedelta(hours=6)).hour.to_numpy()
    med = {(j, h): np.nanmedian(pm[:6552, j][hour[:6552] == h])
           for j in range(len(SITES)) for h in range(24)}
    return np.array([med[j, hour[t]] for t, j, _ in rows])

