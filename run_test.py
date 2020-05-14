#!/usr/bin/env python3

"""
Little script by chocorion.
"""

import subprocess, sys, os
from subprocess import PIPE

NUMBER_TRY = 4
TIMEOUT_SECONDS = 390

OPENCL_PLATFORM = 0

VARIANTS = [
    {
        'name': 'seq',
        'options': []
    },
   
    {
        'name': 'ocl_sync',
        'options': ['-o'],
        'env' : {
            'TILEX': '16',
            'TILEY': '16'
        }

    },
    {
        'name': 'ocl_tiled',
        'options': ['-o'],
        'env' : {
            'TILEX': '16',
            'TILEY': '16'
        }

    },

]

ARGS = [
     {
        'name': 'alea',
        'size': ['960', '1920', '3840']
    }
]

USE_DUMP = True

def clean_png():
    subprocess.run(
        ['rm *.png'],
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

if __name__ == "__main__":
    clean_png()

    for arg in ARGS:
        for size in arg['size']:

            ref_time = -1
            for variant in VARIANTS:
                cmd = ['./run', '-v', variant['name'], '-a', arg['name'], '-s', size, '-k', 'sable', '-n'] + [i for i in variant['options']]

                if USE_DUMP:
                    cmd += ['-du']
                
                time_sum = 0.
                correct = True

                env_dict = {}

                if '-o' in variant['options']:
                    env_dict["PLATFORM"] = str(OPENCL_PLATFORM)

                    if 'env' in variant.keys():
                        env_dict = variant['env']

                        # Ugly
                        if "TILEX" in variant['env'].keys():
                            cmd += ['-g', str(int(size)//int(variant['env']['TILEX']))]

                    elif '-g' in variant['options']:
                        grain_index = variant['options'].index('-g') + 1
                        grain = int(variant['options'][grain_index])

                        env_dict["TILEX"] = str(int(int(size)/grain))
                        env_dict["TILEY"] = str(int(int(size)/grain))
                    
                for i in range(NUMBER_TRY):
 
                    p = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE, env=dict(os.environ, **env_dict))  

                    try :
                        outs, errs = p.communicate(timeout=TIMEOUT_SECONDS)
 
                        time_index = len(errs.decode('utf-8').split('\n')) - 2
                        time = float(errs.decode('utf-8').split('\n')[time_index])

                    except subprocess.TimeoutExpired:
                        p.kill()
                        correct=False
                        print("\033[31mTIMEOUT for cmd -> {}\033[0m".format(' '.join(cmd)))
                        break

                    except ValueError:
                        correct=False
                        print("\033[31mValue error, probably an error in response parse. Get :\n{}\n\033[0m".format(errs.decode('utf-8')))
                        break
                    except:
                        print("\033[31mError for cmd -> {}, {}\033[0m".format(' '.join(cmd), sys.exc_info()[0]))
                        correct = False
                        break

                    time_sum += time
                
                moyenne = time_sum/NUMBER_TRY

                if correct:
                    CORRECT_OUTPUT = True

                    # First time is used as reference
                    if ref_time == -1:
                        ref_time = moyenne

                    else:
                        if USE_DUMP:
                            p = subprocess.run(
                                ['diff dump-sable-{}-dim-{}-iter-*.png'.format('seq', size) + ' dump-sable-{}-dim-{}-iter-*.png'.format(variant['name'], size)],
                                shell=True,
                                stdout=PIPE,
                                stderr=PIPE
                            )

                            if len(p.stdout.decode('utf-8')) != 0:
                                CORRECT_OUTPUT = False

                            p = subprocess.run(
                                ['rm dump-sable-{}-dim-{}-iter-*.png'.format(variant['name'], size)],
                                shell=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )

                    env_str = ""
                    if 'env' in variant.keys():
                        for key in variant['env'].keys():
                            env_str += "{}={} ".format(key, variant['env'][key])
                            
                    print("{:80s} --> {:8s} Speedup : ".format(env_str + ' '.join(cmd), "{:.2f}".format(moyenne)), end="")
                    speedup = ref_time/moyenne

                    if speedup >= 1.:
                        print("\t\033[92m{:.2f}\033[0m".format(speedup), end="")
                    else:
                        print("\t\033[91m{:.2f}\033[0m".format(speedup), end="")

                    if USE_DUMP:
                        if CORRECT_OUTPUT:
                            print("\t\033[92mcorrect   output\033[0m")
                        else:
                            print("\t\033[91mincorrect output\033[0m")
                    else:
                        print("\t\033[34mno output verification\033[0m")

            print('')
            clean_png()
