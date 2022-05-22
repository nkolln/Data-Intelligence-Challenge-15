There are two portions to the code and the code is written in such a way that each can be run independently from the other. In other words, if you need to
save time you can e.g. only run the main testing code and forego the hyperparameter search. The way to run each portion is described below, all code should 
be run in the command line (Anaconda prompt) from directory ./Discrete-Simulations/

Hyperparameter search (takes about 2.5h total):
1) run 'python headless.py hyperparameter-search'
2) run 'python MC_test.py hyperparameter-search'
3) run 'python plotting.py hyperparameters' (this prints the best hyperparameter combinations found)

Generating main results (takes about 45 min total):
1) run 'python headless.py baseline'
2) run 'python headless.py QL-SARSA-test'
3) run 'python MC_test.py test'
4) run 'python plotting.py main-results' (this generates and saves as .png files the figures included in the report)

IMPORTANT NOTE on multiprocessing: By default headless.py works on 4 cores and MC_test.py on 3. If you want a different number of cores, you can specify
this by running the above commands with the number of cores as an extra argument. E.g. 'python headless.py baseline 2' will run on 2 cores. Do beware that
the time estimates given are with the default number of cores, less cores may slow them down.

Note on libraries: tqdm and pandas are needed (also added to requirements.txt).