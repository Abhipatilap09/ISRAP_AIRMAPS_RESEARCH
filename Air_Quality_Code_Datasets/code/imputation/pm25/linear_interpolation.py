"""Observed-endpoint interpolation used in the PM2.5 comparisons."""
import numpy as np

def interpolate(pm_masked, rows):
    ans = []
    for t, j, gap in rows:
        left, right = t-1, t+1
        while left >= 0 and not np.isfinite(pm_masked[left, j]): left -= 1
        while right < len(pm_masked) and not np.isfinite(pm_masked[right, j]): right += 1
        if left >= 0 and right < len(pm_masked) and right-left <= gap+1:
            ans.append(pm_masked[left, j] + (pm_masked[right, j]-pm_masked[left, j])*(t-left)/(right-left))
        else:
            ans.append(np.nan)
    return np.asarray(ans)

