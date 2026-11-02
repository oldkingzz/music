"""rD_hist_m1_h0_v1 (Phase D history ablation): m1_filter_v5lo with its HMM chord filter limited to her last 0 beats (H = 0)."""
from systems import m1_filter_v5lo

H = 0   # beats of her past melody the HMM filter may see (m1_filter_v5lo itself = the whole song so far)


def build(train):
    f = m1_filter_v5lo.build(train)
    f.H = H   # Filter.belief then reruns the HMM filter over only the last H beats (what DUET_HIST did)
    return f
