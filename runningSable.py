#!/usr/bin/env python3
from graphTools import *
from expTools import *
import os

options = {}
ompenv  = {}
#Dictionnaire avec les options de compilations d'apres commande
options["-k "] = ["sable"]
options["-s "] = [128, 256, 512, 1024]
ompenv["OMP_NUM_THREADS="] = [1]

options["-v "] = ["seq"]
execute('./bin/easypap', ompenv, options)

options["-v "] = ["vec"]
execute('./bin/easypap', ompenv, options)

#Dictionnaire avec les options OMP
ompenv["OMP_NUM_THREADS="] = [1,2,4,8]
ompenv["OMP_PLACES="] =["cores","threads"]

options["-v "] = ["ompfor"]
execute('./bin/easypap', ompenv, options)
