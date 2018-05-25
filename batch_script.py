import os

# try blocks to enable stopping currently running command
# without stopping this script
# TODO: compare with other algorithms
try:
    os.system("python desa_experiment.py -d 40")
except:
    pass

try:
    os.system("python desa_experiment.py -d 40 differential_evolution")
except:
    pass

try:
    os.system("python desa_experiment.py -d 40 random_search")
except:
    pass

# check parameters influence on algorithm performance
# maybe it should be checked only on one function 
# in bbob test suite, eg. Rastrigin

# reference with default parameters
try:
    os.system("python desa_experiment.py -d 10")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -m 1.5")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -m 0.5")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -c 1")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -m 0.5")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -b 1e3")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -b 5e3")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -p 10")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -p 40")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -t 100000")
except:
    pass

try:
    os.system("python desa_experiment.py -d 10 -t 0")
except:
    pass

# TODO: check strategy influence
