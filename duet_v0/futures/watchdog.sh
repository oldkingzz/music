#!/bin/bash
# Watchdog for the futures run (bash 3.2): prints a line only when something needs the lead's attention.
F=/Users/wangzesen/work/claude_sandbox/MEAM5160/project/duet_v0/futures
SEEN=$F/status/.wd_seen; mkdir -p "$SEEN"
START=${START:-$(date +%s)}
say() { k="$1"; shift; if [ ! -e "$SEEN/$k" ]; then : > "$SEEN/$k"; echo "$(date +%H:%M) $*"; fi; }
while true; do
  now=$(date +%s); hm=$(date +%H%M)
  for id in d1 d2 d3 d4 d5; do
    s="$F/status/$id.json"
    if [ -f "$s" ]; then
      age=$(( now - $(stat -f %m "$s") ))
      if grep -q '"done": *true' "$s"; then say "done-$id" "DONE $id builder"
      else
        [ $age -gt 600 ] && say "stale-$id-$(( now / 300 ))" "STALE $id heartbeat ${age}s old: $(tr -d '\n' < "$s" | cut -c1-160)"
        [ "$hm" -ge 1156 ] && say "over-$id" "OVERRUN $id builder past 11:55, not done"
      fi
    elif [ $(( now - START )) -gt 780 ]; then say "nostatus-$id-$(( now / 300 ))" "NOSTATUS $id: no heartbeat file after $(( (now-START)/60 )) min"; fi
    v="$F/status/verify_$id.json"
    if [ -f "$v" ]; then
      if grep -q '"done": *true' "$v"; then say "vdone-$id" "DONE $id verifier"
      else
        vage=$(( now - $(stat -f %m "$v") ))
        [ $vage -gt 600 ] && say "vstale-$id-$(( now / 300 ))" "STALE verify_$id heartbeat ${vage}s old"
        [ "$hm" -ge 1213 ] && say "vover-$id" "OVERRUN verify_$id past 12:12"
      fi
    fi
  done
  free=$(df -k / | awk 'NR==2{print $4}')
  [ "$free" -lt 1048576 ] && say "disk-$(( now / 600 ))" "DISK below 1 GB free: $((free/1024)) MB"
  ps -axo pid=,etime=,command= | awk '/python/ && /(duet_v0|futures)/ && !/watchdog/ {n=split($2,a,/[-:]/); s=0; if(n==2)s=a[1]*60+a[2]; else if(n==3)s=a[1]*3600+a[2]*60+a[3]; else if(n==4)s=a[1]*86400+a[2]*3600+a[3]*60+a[4]; if(s>540) print $1, s, substr($0, index($0,$3), 140)}' | while read pid et cmd; do
    say "proc-$pid" "LONGPROC pid $pid running ${et}s: $cmd"
  done
  sleep 60
done
