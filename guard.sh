#!/bin/zsh
# Stop the local trial if the Mac runs short of disk or starts swapping
# hard. Usage: ./guard.sh <pid-of-run_local.sh>
PID=$1
while kill -0 $PID 2>/dev/null; do
  free_gb=$(df -g /System/Volumes/Data | awk 'NR==2 {print $4}')
  swap_mb=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  if (( free_gb < 8 )) || (( ${swap_mb%.*} > 16000 )); then
    echo "$(date) guard: free ${free_gb} GB, swap ${swap_mb} MB, stopping" >&2
    pkill -P $PID; kill $PID
    exit 1
  fi
  sleep 30
done
