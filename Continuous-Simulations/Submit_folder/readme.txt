Imports that are not baseline
 - Geopandas (pip install geopandas)
 - Shapely (pip install shapely)
 - Numpy (pip install numpy)

 The code is split into a couple run files, an environment file, and an action controller file.
  - pygame_env.py: This is the file that controls the environment and also sets rewards
  - cont_act_control_v4.py: This file contains the logic behind how the scores are scored and how the final vector is chosen.
  - cont_action_base.py: This file contains a random map and can be queried using mode -2 for the greedy robot and -1 for the Q-learning robot.
  - cont_action_greedy_loop.py: This file contains the loops that goes through all the different rooms for the greedy bot. This is set to run once through everything and can be used for testing.
  - cont_action_Q_loop.py: This file contains the loop that goes through all the different rooms for the Q learning. Is set to run once, run through for testing.