#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

options = {}
ompenv  = {}
nbrun   = 3
# Configuration global
options["-k "] = ["sable"]
options["-s "] = [480]
options["-a "] = ["4partout"]
# Configuration des versions mono-thread
ompenv["OMP_NUM_THREADS="]  = [1]
ompenv["OMP_PLACES="]       = ["cores"]
ompenv["OMP_SCHEDULE="]     = ["static"]

options["-v "] = ["seq"]
execute('./bin/easypap', ompenv, options, nbrun)

options["-v "] = ["vec", "vec2"]
execute('./bin/easypap', ompenv, options, nbrun)

# Configuration des versions multi-threads
ompenv["OMP_NUM_THREADS="]  = [1,2,4,8,12,16,24,48]
ompenv["OMP_PLACES="]       = ["cores","threads"]
ompenv["OMP_SCHEDULE="]     = ["static", "dynamic", "guided"]

options["-v "] = [
                    "ompfor", "ompfor2",
                    "ompfor_tiled", "ompfor_tiled2",
                    "vec_ompfor", "vec_ompfor2",
                    "vec_ompfor_tiled", "vec_ompfor_tiled2"
                ]
execute('./bin/easypap', ompenv, options, nbrun)