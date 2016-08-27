#!/bin/sh

PID=$(cat pid)
CPIDS=$(pgrep -P $PID)
#kill -TERM $CPIDS
echo Finishing process...
kill -9 $PID
echo Done

