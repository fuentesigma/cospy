#!/bin/bash


if [ -z "$1" ]
then
  echo "Usage: `basename $0` <<parameter file>>"
  echo "example: `basename $0` input.par"
  exit $E_BADARGS
fi  

resolution="32 64 128 256"

for N in $resolution
	do
	cmd="cospy_cmd.py $1 -nx $N --project_directory data_$N"
	echo "running comand: $cmd"
	`$cmd`
	echo "done"
	done
