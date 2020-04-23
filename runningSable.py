#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

options = {}
ompenv  = {}
nbrun   = 3
#Dictionnaire avec les options de compilations d'apres commande
options["-k "] = ["sable"]
options["-s "] = [256, 512, 1024]
ompenv["OMP_NUM_THREADS="] = [1]
ompenv["OMP_PLACES="] = ["cores"]
ompenv["OMP_SCHEDULE="] = ["static"]

options["-v "] = ["seq"]
execute('./bin/easypap', ompenv, options, 3)

options["-v "] = ["vec", "vec2"]
execute('./bin/easypap', ompenv, options, 3)

#Dictionnaire avec les options OMP
ompenv["OMP_NUM_THREADS="] = [1,2,4,8]
ompenv["OMP_PLACES="] = ["cores","threads"]
ompenv["OMP_SCHEDULE="] = ["static", "dynamic", "guided"]

options["-v "] = ["ompfor", "ompfor2", "vec_ompfor", "vec_ompfor2"]
execute('./bin/easypap', ompenv, options, 3)
