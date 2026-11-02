#!/usr/bin/env python3
"""Duet live: the real-time application that runs on the MacBook.

    # 1. rehearsal: analyse the reference once (features, notes, harmony plan, breaths, beat)
    python live/duet_live.py rehearse song.wav -o live/cache/song [--take rehearsal_take.wav]

    # 2. perform
    python live/duet_live.py run --ref live/cache/song --backend file --input take.wav [--fast] [--view]
    python live/duet_live.py run --ref live/cache/song --backend mac [--view]          # mic + speaker
    python live/duet_live.py run --ref live/cache/song --backend device --port /dev/cu.usbmodem1101

Every run writes into live/runs/<name>/ : duet.wav (file mode: L = singer, R = machine),
log.jsonl (one record per 10 ms frame + one per event), summary.json.
See live/README.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

LIVE = Path(__file__).resolve().parent
ROOT = LIVE.parent
for p in (str(ROOT), str(LIVE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from engine import Engine, JsonlLogger  # noqa: E402
from dsp import SR  # noqa: E402
from rehearsal import Reference, choose_plan, load_audio, parse_bias, rehearse  # noqa: E402


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def gate_arg(v):
    """--gate-db: dB below the reference's 90th-percentile frame level, or 'off'."""
    if str(v).lower() in ('off', 'none'):
        return 'off'
    return float(v)


def resolve_front_end(ref, args):
    """The evaluation's front end for this run: input gate threshold (dBFS or None) and detector bias.

    Gate: --gate-db off -> none; --gate-db X -> threshold = the reference's 90th-percentile frame level - X
    (eval/duet_eval/features.gate_threshold); not given -> the gate the reference was rehearsed with
    (`rehearse --gate-db`), i.e. none for every cache made without it. Bias: --bias overrides the
    cache's bias_live for this run."""
    info = {'gate': None, 'bias': {'used': ref.bias_live, 'source': 'rehearsal cache'}}
    want = getattr(args, 'gate_db', None)
    thr = None
    if want is None and ref.gate is not None:
        thr = ref.gate['thr_db']
        info['gate'] = dict(ref.gate, from_='rehearsal cache')
    elif want not in (None, 'off'):
        if ref.gate is not None:
            level = ref.gate['level_p90_db']
            if abs(want - ref.gate['gate_db']) > 1e-9:
                print(f"input gate: {want:g} dB at run time, but the reference was analysed with a "
                      f"{ref.gate['gate_db']:g} dB gate (the evaluation uses one gate for both)", flush=True)
        else:
            from rehearsal import eval_features
            src = ref.meta.get('reference')
            if not src or not Path(src).exists():
                sys.exit('--gate-db: this cache has no stored reference level and its reference recording is gone; '
                         're-run rehearse with --gate-db')
            level = float(np.percentile(eval_features().frame_db(load_audio(src)), 90))
            print('input gate: the reference was rehearsed WITHOUT a gate; the evaluation gates both takes '
                  '(re-run rehearse with --gate-db for the same front end)', flush=True)
        thr = level - want
        info['gate'] = {'gate_db': want, 'level_p90_db': level, 'thr_db': thr, 'from_': 'command line',
                        'source': 'eval/duet_eval/features.py'}
    if getattr(args, 'bias', None):
        ref.bias_live, src = parse_bias(args.bias)
        info['bias'] = {'used': ref.bias_live, 'source': src}
        if ref.profile_grid is not None:
            print('bias: --bias at run time does not re-learn the rehearsal timing profile '
                  '(give --bias to rehearse for that)', flush=True)
    return thr, info


def build_engine(args, logger):
    """Reference + harmony plan + echo guard + MIDI out -> Engine (shared by every backend)."""
    ref = Reference.load(args.ref)
    choose_plan(ref, args.harmony)
    gate_thr, args.front_end = resolve_front_end(ref, args)
    midi = None
    if args.midi_out or args.midi_connect:
        from midi import MidiOut
        midi = MidiOut(name=args.midi_out or 'Duet', connect=args.midi_connect, channel=args.midi_channel - 1,
                       midi_latency=args.midi_latency, in_latency=args.in_latency or 0.0,
                       nod_cc=None if args.nod_cc < 0 else args.nod_cc)
        print(f'MIDI out: {midi.port_desc}, channel {args.midi_channel}, latency compensation '
              f'{args.midi_latency * 1000:.0f} ms', flush=True)
    synth = args.synth == 'on' or (args.synth == 'auto' and (midi is None or args.backend == 'file'))
    guard = None
    if args.echo_guard:
        from echo_link import make_guard
        aec = args.echo_mode in ('full', 'aec')
        mask = args.echo_mode in ('full', 'mask')
        if aec and not synth:
            print('echo guard: the machine voice comes from an external synth (MIDI), so the canceller has no '
                  'exact reference; running the pitch mask only (use headphones for best results)', flush=True)
            aec = False
        elif aec and args.soundcheck <= 0:
            print('echo guard: no sound check -> canceller OFF (a canceller started cold during the song learns '
                  'to cancel the singer; echo/README.md); pitch mask only', flush=True)
            aec = False
        if aec or mask:
            guard = make_guard(aec=aec, mask=mask, freeze_after_soundcheck=args.aec_update == 'frozen')
            if not synth and guard.pm is not None:      # Logic sounds at stream time `at`: ref -> mic = L + room
                guard.pm.delay = int(round((args.out_latency + 0.002) * SR))
    eng = Engine(ref, out_latency=args.out_latency, compute_margin=args.compute_margin, conf_thr=args.conf_thr,
                 nod=args.nod, use_profile=not args.no_profile, logger=logger, log_every=args.log_every,
                 guard=not args.no_guard, echo=guard, midi=midi, synth=synth, gate_thr=gate_thr)
    h = ref.meta['harmony']
    g = args.front_end['gate']
    print('input gate: ' + ('off' if g is None else f"{g['gate_db']:g} dB below the reference p90 level "
                                                     f"(thr {g['thr_db']:.1f} dBFS, {g['from_']})")
          + f"; detector bias: {args.front_end['bias']['source']}", flush=True)
    print(f"harmony: {h['used']} ({h['key']}, {h['n_plan']} notes)"
          + (f"; echo guard: {'canceller+' if guard.aec is not None else ''}{'mask' if guard.pm is not None else ''}"
             if guard is not None else '; echo guard: off')
          + f"; out latency {args.out_latency * 1000:.0f} ms" + ('' if synth else '; built-in voice muted'), flush=True)
    return eng


def cmd_rehearse(args):
    prefix = args.out or str(LIVE / 'cache' / Path(args.reference).stem)
    bias, src = parse_bias(args.bias) if args.bias else (None, None)
    rehearse(args.reference, prefix, take=args.take, plan_json=args.plan, join_at=args.join_at, beat=args.beat,
             notes_file=args.notes, gate_db=args.gate_db, bias=bias, bias_source=src)


def resolve_args(args):
    """Backend-dependent defaults."""
    if args.out_latency is None:
        args.out_latency = {'file': 0.005, 'mac': 0.020, 'device': 0.052}[args.backend]
    if args.echo_guard is None:                 # ON for mic + speaker in the same room; file: only with bleed
        args.echo_guard = args.backend in ('mac', 'device') or args.simulate_bleed is not None
    if args.simulate_bleed is not None and args.backend != 'file':
        sys.exit('--simulate-bleed is for --backend file (a simulated room); the real backends have a real one')
    if args.in_latency is None:
        args.in_latency = {'file': 0.0, 'mac': None, 'device': 0.003}[args.backend]
    if args.midi_out or args.midi_connect:
        if args.backend == 'file' and args.fast:
            sys.exit('--midi-out needs real time (drop --fast)')
        need = (args.in_latency or 0.0) + args.midi_latency + 0.008
        if args.out_latency < need:
            print(f'MIDI: output latency raised {args.out_latency * 1000:.0f} -> {need * 1000:.0f} ms so every note is '
                  f'decided before it must be sent (prediction makes this free; see README)', flush=True)
            args.out_latency = need
    return args


def cmd_run(args):
    sys.setswitchinterval(0.001)       # GIL hand-off every 1 ms (default 5 ms): log/view threads stay short
    resolve_args(args)
    name = args.name or f'{args.backend}_{Path(args.input).stem if args.input else Path(args.ref).stem}_{time.strftime("%Y%m%d-%H%M%S")}'
    run_dir = Path(args.run_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_dir / 'log.jsonl') if not args.no_log else None
    engine = build_engine(args, logger)
    view = None
    if args.view is not None:
        from viewserver import ViewServer
        view = ViewServer(engine, port=args.view, extra={'backend': args.backend}).start()
        print(f'live view: http://127.0.0.1:{view.port}/   (state: /state.json)', flush=True)
        if args.view_wait:
            time.sleep(args.view_wait)
    t0 = time.time()
    try:
        if args.backend == 'file':
            from backends import FileBackend
            if not args.input:
                sys.exit('--input is required for --backend file')
            x = load_audio(args.input)
            bleed = None
            if args.simulate_bleed is not None:
                from echo_link import BleedSim
                bleed = BleedSim(x, args.simulate_bleed, acoustic_ms=args.bleed_acoustic_ms, rt60=args.bleed_rt60)
                print(f'simulated bleed: {args.simulate_bleed:+.1f} dB re her voice at the mic (echo/sim.py room, '
                      f'{args.bleed_acoustic_ms:.1f} ms acoustic delay, RT60 {args.bleed_rt60:.2f} s)', flush=True)
            be = FileBackend(engine, x, run_dir / 'duet.wav', realtime=not args.fast, tail=args.tail, monitor=args.monitor,
                             bleed=bleed, soundcheck_s=args.soundcheck)
            _, bres = be.run()
        elif args.backend == 'mac':
            from backends import MacBackend
            be = MacBackend(engine, device=args.audio_device, latency=args.latency, duration=args.duration,
                            record_path=run_dir / 'session.wav',
                            soundcheck_s=args.soundcheck if engine.synth else 0.0)
            bres = be.run()
        else:
            from backends import DeviceBackend
            if not args.port:
                sys.exit('--port is required for --backend device (e.g. /dev/cu.usbmodem1101)')
            be = DeviceBackend(engine, args.port, record_path=run_dir / 'session.wav', prefill=args.spk_prefill,
                               keep_level=not args.no_level_keeper,
                               soundcheck_s=args.soundcheck if engine.synth else 0.0, in_latency=args.in_latency)
            bres = be.run(max_frames=int(args.duration * 100) if args.duration else None)
    finally:
        if engine.midi is not None:
            engine.midi.close()
    if view is not None and args.view_hold:
        print(f'run finished; holding the view for {args.view_hold} s', flush=True)
        time.sleep(args.view_hold)
    if view is not None:
        view.stop()
    if logger:
        logger.close()
    summ = engine.summary()
    out = {'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != 'func'},
           'front_end': args.front_end,
           'backend': bres, 'engine': summ, 'reference': engine.ref.meta,
           'notes_out': sorted(engine.out_notes.values(), key=lambda q: q['at']),
           'decisions': {j: {'status': v['status'], 'at': v.get('at')} for j, v in sorted(engine.acc.pred.items())},
           'taken_back': engine.taken_back, 'suppressed': engine.suppressed, 'held': [e for e in engine.events if e['ev'] in ('held', 'missed')],
           'nods': engine.nod.log, 'wall_s': time.time() - t0, 'run_dir': str(run_dir),
           'plan': engine.acc.plan, 'key': engine.ref.key,
           'midi_log': None if engine.midi is None else engine.midi.log}
    (run_dir / 'summary.json').write_text(json.dumps(_jsonable(out), indent=1))
    fm = summ['frame_ms']
    if summ.get('echo_guard'):
        g = summ['echo_guard']
        sc = bres.get('soundcheck') if isinstance(bres, dict) else None
        print('echo guard: ' + ', '.join(f'{k} {v}' for k, v in g.items() if v is not None)
              + (f" | sound check {sc}" if sc else ''), flush=True)
    print(f"{args.backend}: {summ['audio_s']:.1f} s audio, frame p50 {fm['p50']:.3f} ms p99 {fm['p99']:.3f} ms "
          f"max {fm['max']:.3f} ms | notes played {summ['notes']['played']} held {summ['notes']['held']} "
          f"missed {summ['notes']['missed']} taken back {summ['notes']['taken_back']} | nods {len(summ['nods'])}"
          + (f" | RTF {bres['rtf']:.4f}" if isinstance(bres, dict) and 'rtf' in bres else '')
          + f'\n-> {run_dir}', flush=True)
    return out


def cmd_devices(args):
    import sounddevice as sd
    print(sd.query_devices())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('rehearse', help='analyse a reference recording and cache it')
    r.add_argument('reference', help='reference recording (any sample rate; mixed to mono, resampled to 24 kHz)')
    r.add_argument('-o', '--out', help='cache prefix (default live/cache/<stem>) -> <prefix>.npz + <prefix>.json')
    r.add_argument('--take', help='a rehearsal take of the singer: learn her timing profile from it')
    r.add_argument('--notes', help='annotated notes of the reference (rows: onset offset midi) instead of '
                                   'automatic segmentation; recommended for real singing (see README)')
    r.add_argument('--plan', help='harmony plan JSON {"plan": [...], "key": ...} instead of the rule-based one')
    r.add_argument('--join-at', type=float, help='machine joins at this fraction (<=1) or second (>1) of the song')
    r.add_argument('--beat', type=float, help='beat length in reference seconds (default: estimated)')
    r.add_argument('--gate-db', type=float, metavar='DB',
                   help="the evaluation's input level gate (eval/duet_eval/features.py; default OFF): frames more "
                        "than ~DB below the reference's 90th-percentile frame level are faded out (6 dB soft knee). "
                        "Fixed here, applied to the reference, the --take and (via the cache) the live input. "
                        "The evaluation used 15 on choir close mics (other singers' bleed); see README")
    r.add_argument('--bias', metavar='A,G,R|DATASET',
                   help="detector bias (s, live anchor - true onset) for attack,glide,repeat onsets instead of the "
                        "synthetic-singer calibration, or a dataset name ('dcs', 'ecd') to use the evaluation's "
                        "per-dataset values from eval/results/v2_frozen.json")
    r.set_defaults(func=cmd_rehearse)

    p = sub.add_parser('run', help='perform live')
    p.add_argument('--ref', required=True, help='cache prefix written by rehearse')
    p.add_argument('--backend', choices=('file', 'mac', 'device'), default='file')
    p.add_argument('--input', help='file backend: the performance to play in')
    p.add_argument('--fast', action='store_true', help='file backend: as fast as possible instead of real time')
    p.add_argument('--tail', type=float, default=0.5, help='file backend: seconds of silence appended')
    p.add_argument('--monitor', action='store_true', help='file backend: also play the duet on the speakers')
    p.add_argument('--port', help='device backend: serial port of the ESP32-S3')
    p.add_argument('--audio-device', help='mac backend: sounddevice device name/index (default system default)')
    p.add_argument('--latency', default='low', help="mac backend: PortAudio latency ('low', 'high' or seconds)")
    p.add_argument('--duration', type=float, help='mac/device: stop after this many seconds')
    p.add_argument('--out-latency', type=float,
                   help='output latency L in s (default file 0.005, mac measured from PortAudio, device 0.052 = '
                        '4-frame jitter buffer + 10.65 ms DMA/filter + USB)')
    p.add_argument('--spk-prefill', type=int, default=4,
                   help='device: silent frames sent before the first answer = speaker jitter buffer depth (default 4)')
    p.add_argument('--no-level-keeper', action='store_true',
                   help='device: do not correct the speaker buffer level after an underrun (see README)')
    p.add_argument('--harmony', choices=('learned', 'rule', 'cache'),
                   help="harmony line: 'learned' (harmony/model.py CRF; default if it loads), 'rule' (3rd/6th "
                        "below), 'cache' (the plan stored at rehearsal, e.g. from --plan)")
    p.add_argument('--echo-guard', action=argparse.BooleanOptionalAction, default=None,
                   help='echo canceller + pitch mask against speaker->mic bleed (echo/guard.py). '
                        'Default ON for mac/device, and for file mode when --simulate-bleed is given')
    p.add_argument('--echo-mode', choices=('full', 'aec', 'mask'), default='full',
                   help='which parts of the guard to run (default both)')
    p.add_argument('--aec-update', choices=('frozen', 'adaptive'), default='frozen',
                   help="after the sound check keep the canceller's filter fixed (default) or let echo/guard.py "
                        "replace it during the song (see README: 'adaptive' lost the pause scenario at >= 0 dB)")
    p.add_argument('--soundcheck', type=float, default=2.0,
                   help='seconds of soft pink noise before the song to train the canceller (0 = skip: canceller off)')
    p.add_argument('--simulate-bleed', type=float, metavar='DB',
                   help='file mode: mix the machine voice back into the input through the echo/ room model, '
                        'DB = echo level re her voice at the mic (e.g. -6)')
    p.add_argument('--bleed-acoustic-ms', type=float, default=2.0, help='simulated speaker->mic distance as delay')
    p.add_argument('--bleed-rt60', type=float, default=0.3, help='simulated room RT60 (s)')
    p.add_argument('--midi-out', nargs='?', const='Duet', metavar='NAME',
                   help='send the machine voice as MIDI on a virtual CoreMIDI output named NAME (default "Duet"), '
                        'e.g. into Logic Pro (see live/LOGIC.md)')
    p.add_argument('--midi-connect', metavar='DEST',
                   help='send MIDI to an existing destination whose name contains DEST (e.g. "IAC") instead')
    p.add_argument('--midi-latency', type=float, default=0.010,
                   help='synth MIDI-in -> sound latency to compensate (s, default 0.010; measure it, LOGIC.md)')
    p.add_argument('--midi-channel', type=int, default=1)
    p.add_argument('--nod-cc', type=int, default=20, help='CC number that mirrors the servo nod (-1 = none)')
    p.add_argument('--in-latency', type=float,
                   help='capture -> host arrival latency for the MIDI clock (default file 0, mac PortAudio, device 0.003)')
    p.add_argument('--synth', choices=('auto', 'on', 'off'), default='auto',
                   help="built-in voice: 'auto' = muted when MIDI drives an external synth (except file mode)")
    p.add_argument('--compute-margin', type=float, default=0.0, help="extra s passed as Accompanist 'compute'")
    p.add_argument('--conf-thr', type=float, default=0.55)
    p.add_argument('--nod', choices=('first', 'phrases', 'off'), default='first',
                   help='servo nod before the first entry, before every phrase entry, or never')
    p.add_argument('--no-profile', action='store_true', help='ignore the rehearsal timing profile')
    p.add_argument('--gate-db', type=gate_arg, metavar='DB|off',
                   help="input level gate (eval/duet_eval/features.StreamGate): DB below the reference's "
                        "90th-percentile frame level, or 'off'. Default: whatever `rehearse --gate-db` stored "
                        "(off for a cache rehearsed without it)")
    p.add_argument('--bias', metavar='A,G,R|DATASET',
                   help="override the cache's detector bias for this run (as rehearse --bias)")
    p.add_argument('--no-guard', action='store_true',
                   help='render every Accompanist decision, even two notes at the same instant (= run_v0)')
    p.add_argument('--view', type=int, nargs='?', const=8765, help='serve the live view on this port (default 8765)')
    p.add_argument('--view-wait', type=float, default=0.0, help='seconds to wait after starting the view')
    p.add_argument('--view-hold', type=float, default=0.0, help='keep the view up this long after the run')
    p.add_argument('--log-every', type=int, default=1, help='write every Nth frame record to log.jsonl')
    p.add_argument('--no-log', action='store_true')
    p.add_argument('--run-dir', default=str(LIVE / 'runs'))
    p.add_argument('--name', help='run folder name (default backend_input_timestamp)')
    p.set_defaults(func=cmd_run)

    d = sub.add_parser('devices', help='list audio devices (does not open them)')
    d.set_defaults(func=cmd_devices)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    main()
