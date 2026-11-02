"""Physical design: how much machine voice reaches the mic for each mic placement.

Back-of-envelope acoustics (free-field point sources + diffuse reverberant field), in dB SPL:

  direct level at distance r      L(r) = L(1 m) - 20 log10(r / 1 m)         (1/r law, -6 dB per doubling)
  reverberant level (anywhere)    L_rev = L(1 m) - 20 log10(r_c / 1 m)      (equals direct at r_c)
  critical distance               r_c = 0.057 sqrt(Q V / RT60)              (V m^3, RT60 s, Q directivity)
  first-order mic, angle theta    g(theta) = a + (1 - a) cos(theta); cardioid a = 0.5
  diffuse-field pickup            random energy efficiency, cardioid 0.333 (-4.8 dB)

The mic is assumed to point at the singer. The speaker's direct sound arrives from angle theta,
the room's reverberant echo from everywhere (so a directional mic only gains its REE on it).
A real cardioid capsule does not have the ideal -inf null at 180 deg; we assume 15 dB
(REAR_REJECTION_DB), which must be checked against the chosen capsule's polar plot.

    .venv/bin/python echo/physical.py     -> echo/results/physical.json (+ printed table)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SINGER_1M_DB = 80.0     # moderate classical singing at 1 m (Hunter, Svec & Titze 2006 report ~82-85 dB loud /a/)
MACHINE_1M_DB = 77.0    # machine voice set ~3 dB under her at 1 m so the harmony sits below the melody
ROOM_V, RT60, Q_SPK = 40.0, 0.3, 2.0
REAR_REJECTION_DB = 15.0    # assumed practical cardioid rejection near 180 deg (ideal: -inf)
REE_CARDIOID_DB = -4.8      # random-energy efficiency of a cardioid
ERLE_AEC_DB = 25.0          # sound-checked canceller, conservative vs. the 21-29 dB measured in sim


def critical_distance(V=ROOM_V, rt60=RT60, q=Q_SPK):
    return 0.057 * np.sqrt(q * V / rt60)


def cardioid_gain_db(theta_deg, rear_floor_db=-REAR_REJECTION_DB):
    g = 0.5 + 0.5 * np.cos(np.radians(theta_deg))
    return max(20 * np.log10(max(g, 1e-6)), rear_floor_db)


def bleed_db(singer_dist, spk_dist, spk_angle_deg, cardioid):
    """Echo level re singer level at the mic (both direct + reverberant), dB."""
    rc = critical_distance()
    s_dir = SINGER_1M_DB - 20 * np.log10(singer_dist)
    s_rev = SINGER_1M_DB - 20 * np.log10(rc)
    m_dir = MACHINE_1M_DB - 20 * np.log10(spk_dist)
    m_rev = MACHINE_1M_DB - 20 * np.log10(rc)
    if cardioid:
        m_dir += cardioid_gain_db(spk_angle_deg)
        m_rev += REE_CARDIOID_DB
        s_rev += REE_CARDIOID_DB
    p = lambda *L: 10 * np.log10(sum(10 ** (x / 10) for x in L))
    return float(p(m_dir, m_rev) - p(s_dir, s_rev)), float(m_dir - m_rev)


CONFIGS = [
    # name, singer->mic m, speaker->mic m, speaker angle re mic axis, cardioid?
    ('A: omni mic on the Duet board, 5 cm from its speaker', 1.0, 0.05, 90, False),
    ('B: cardioid on the far face of the box, 15 cm from speaker (rear)', 1.0, 0.15, 180, True),
    ('C: omni on a stand 20 cm from her mouth, speaker 1 m away', 0.2, 1.0, 180, False),
    ('D: cardioid on a stand 20 cm from her mouth, speaker 1 m behind it', 0.2, 1.0, 180, True),
    ('E: cardioid headset 5 cm from her mouth, speaker 1 m behind', 0.05, 1.0, 180, True),
]


def main():
    rc = critical_distance()
    rows = []
    print(f'critical distance r_c = {rc:.2f} m (V={ROOM_V} m^3, RT60={RT60} s, Q={Q_SPK})')
    for name, ds, dm, ang, card in CONFIGS:
        b, dr = bleed_db(ds, dm, ang, card)
        rows.append({'config': name, 'singer_m': ds, 'speaker_m': dm, 'angle_deg': ang, 'cardioid': card,
                     'bleed_db': round(b, 1), 'bleed_after_aec_db': round(b - ERLE_AEC_DB, 1),
                     'echo_direct_minus_reverb_db': round(dr, 1)})
        print(f'{name:70s} bleed {b:+6.1f} dB   after AEC {b - ERLE_AEC_DB:+6.1f} dB')
    out = {'assumptions': {'singer_1m_db': SINGER_1M_DB, 'machine_1m_db': MACHINE_1M_DB, 'room_m3': ROOM_V,
                           'rt60_s': RT60, 'speaker_Q': Q_SPK, 'critical_distance_m': rc,
                           'cardioid_rear_rejection_db': REAR_REJECTION_DB, 'cardioid_ree_db': REE_CARDIOID_DB,
                           'aec_erle_db': ERLE_AEC_DB},
           'configs': rows}
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'physical.json').write_text(json.dumps(out, indent=1))
    return out


if __name__ == '__main__':
    main()
