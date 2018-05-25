import os

# check parameters influence on algorithm performance
# maybe it should be checked only on one function 
# in bbob test suite, eg. Rastrigin
# TODO: check strategy influence

# reference with default parameters
for f in ['sphere', 'ackeley', 'rosenbrock', 'himmelblau']:
        os.system("python desa_parameter_test.py {}".format(f))

        os.system("python desa_parameter_test.py {} mutation 0.4".format(f))

        os.system("python desa_parameter_test.py {} mutation 1.5".format(f))

        os.system("python desa_parameter_test.py {} crosspoint 0.5".format(f))

        os.system("python desa_parameter_test.py {} crosspoint 1".format(f))

        os.system("python desa_parameter_test.py {} start_temp 100000".format(f))

        os.system("python desa_parameter_test.py {} start_temp 10000".format(f))

        os.system("python desa_parameter_test.py {} start_temp 0".format(f))

        os.system("python desa_parameter_test.py {} alpha 0.2".format(f))

        os.system("python desa_parameter_test.py {} alpha 0.5".format(f))

        os.system("python desa_parameter_test.py {} pop_size 5".format(f))

        os.system("python desa_parameter_test.py {} pop_size 25".format(f))

# try blocks to enable stopping currently running command
# without stopping this script
# TODO?: compare with other algorithms
try:
    os.system("python desa_experiment.py -d 20")
except:
    pass

try:
    os.system("python desa_experiment.py -d 20 differential_evolution")
except:
    pass

try:
    os.system("python desa_experiment.py -d 20 random_search")
except:
    pass
