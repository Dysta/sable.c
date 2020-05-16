#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

options = {}
ompenv  = {}
nbrun   = 3
# Configuration global
options["-k "] = ["sable"]
options["-s "] = [960, 1920, 3840]
options["-a "] = ["alea"]
options["-g "] = [4 ,8, 16, 32, 48]
# Configuration des versions mono-thread
ompenv["OMP_NUM_THREADS="]  = [1]
ompenv["OMP_PLACES="]       = ["cores"]
ompenv["OMP_SCHEDULE="]     = ["static"]

options["-v "] = ["seq"]
execute('./bin/easypap', ompenv, options, nbrun)

ompenv = {}
ompenv[""] = ["TILEX=8 TILEY=8", "TILEX=16 TILEY=16", "TILEX=32 TILEY=32"]
options["-o "] = [""]
options["-v "] = ["ocl_sync", "ocl_sync_freq", "ocl_tiled", "ocl_tiled_freq"]
execute('./bin/easypap', ompenv, options, nbrun)