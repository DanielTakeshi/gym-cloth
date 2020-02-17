"""Combine demonstrator data.

AFTER this, run `analysis/check_demo_data.py` to see a more complete picture of
dataset statistics.
"""
import os
import cv2
import sys
import pickle
import numpy as np
np.set_printoptions(suppress=True, edgeitems=2000, linewidth=200)
from os.path import join

# ---------------------------------------------------------------------------- #
# Pick the data file we want later --- see bottom of this file.                #
# See https://github.com/BerkeleyAutomation/gym-cloth/issues/28 for details.   #
# Arranging data (roughly) chronologically.                                    #
# Format: put the list as first item in tuple, and target file name as second. #
# ---------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------ #
# Outdated, do not use.
# ------------------------------------------------------------------------------------------------ #

# Tier 1, green-ish cloth, no domain randomization, oracle corner exactly.
d_01 = ([
     'demos-2019-08-08-11-03-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
     'demos-2019-08-08-11-04-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
     'demos-2019-08-08-11-05-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
     'demos-2019-08-08-11-05-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
    ],
    'demos-2019-08-08-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
)

# Tier 2, green-ish cloth, no domain randomization, oracle corner exactly.
d_02 = ([
    'demos-2019-08-04-17-57-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-04-17-58-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-04-18-00-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-04-18-01-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    ],
    'demos-2019-08-04-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
)

# Tier 3, green-ish cloth, no domain randomization, oracle corner exactly.
d_03 = ([
    'demos-2019-08-03-14-32-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-03-14-32-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-03-14-32-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-03-14-32-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    ],
    'demos-2019-08-03-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'
)

# ------------------------------------------------------------------------------------------------ #
# Outdated, do not use.
# ------------------------------------------------------------------------------------------------ #

# Tier 1, green-ish cloth, no domain randomization, oracle corner, pulling 'inwards'.
d_04 = ([
    'demos-2019-08-12-16-56-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
    'demos-2019-08-12-16-57-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
    'demos-2019-08-12-16-57-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
    'demos-2019-08-12-16-58-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_500.pkl',
    ],
    'demos-2019-08-12-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
)

# Tier 2, green-ish cloth, no domain randomization, oracle corner, pulling 'inwards'.
d_05 = ([
    'demos-2019-08-13-09-23-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-13-09-24-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-13-09-25-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-13-09-26-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    ],
    'demos-2019-08-13-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
)

# Tier 3, green-ish cloth, no domain randomization, oracle corner, pulling 'inwards'.
d_06 = ([
    'demos-2019-08-12-20-46-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-12-20-48-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-12-20-48-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-12-20-48-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    ],
    'demos-2019-08-12-pol-oracle-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'
)

# ------------------------------------------------------------------------------------------------ #
# Outdated, do not use
# ------------------------------------------------------------------------------------------------ #

# Tier 1, blue-ish cloth, first domain randomization, oracle corner, pulling 'inwards'.
# Took around 250-ish minutes per run, on hermes1.
d_07 = ([
    'demos-2019-08-16-19-43-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-16-19-44-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-16-19-44-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-16-19-44-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-16-19-45-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    ],
    'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
)

# Tier 2, blue-ish cloth, first domain randomization, oracle corner, pulling 'inwards'.
# Took nearly 1200 minutes (!!), ouch 20 hours each??  But now I'm confused why
# was this harder for the oracle policy?? That's one reason it is taking longer.
# NOTE: do not use, seems defective?? Seems like the one I did on the 17th is better for whatever reason.
# That one I generated via Triton4, around 900 minutes for a single 500 episode run.
d_08 = ([
    #'demos-2019-08-16-19-48-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    #'demos-2019-08-16-19-49-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    #'demos-2019-08-16-19-49-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    #'demos-2019-08-16-19-49-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    #'demos-2019-08-16-19-50-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-17-17-20-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-17-17-21-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-17-17-22-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    'demos-2019-08-17-17-25-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_500.pkl',
    ],
    #'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
    'demos-2019-08-17-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
)

# Tier 3, blue-ish cloth, first domain randomization, oracle corner, pulling 'inwards'.
# Took around 800-ish minutes per run, on mason. But confusingly, tier3 is
# easier than tier2 yet for my previous batch of data (before domain
# randomization) it was the opposite?? I'm very confused.
d_09 = ([
    'demos-2019-08-16-19-52-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-16-19-53-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-16-19-53-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-16-19-53-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-16-19-53-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    ],
    'demos-2019-08-16-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'
)


# ------------------------------------------------------------------------------------------------ #
# ORACLE CORNER POLICY, heavier domain randomization 2.0
# ------------------------------------------------------------------------------------------------ #

# Tier 1, blue-ish cloth, second (heavier) domain randomization, oracle corner, pulling 'inwards'.
d_10 = ([
    'demos-2019-08-20-09-01-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-20-09-03-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-20-09-03-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-20-09-04-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-20-09-04-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    ],
    'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl'
)

# Tier 2, blue-ish cloth, second (heavier) domain randomization, oracle corner, pulling 'inwards'.
d_11 = ([
    'demos-2019-08-20-18-19-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-20-18-20-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-20-18-21-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-20-18-21-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-20-18-22-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    ],
    'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl'
)

# Tier 3, blue-ish cloth, second (heavier) domain randomization, oracle corner, pulling 'inwards'.
d_12 = ([
    'demos-2019-08-20-18-25-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-20-18-26-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-20-18-26-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-20-18-26-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-20-18-27-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    ],
    'demos-2019-08-20-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl'
)

# ------------------------------------------------------------------------------------------------ #
# RANDOM POLICY BASELINE (with heavier domain randomization 2.0 but that's not as relevant)
# ------------------------------------------------------------------------------------------------ #

# Tier 1, random policy (triton4).
d_13 = ([
    'demos-2019-08-22-16-35-pol-random-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-16-36-pol-random-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-16-36-pol-random-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-16-36-pol-random-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-16-36-pol-random-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    ],
    'demos-2019-08-22-pol-random-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl',
)

# Tier 2, random policy (triton4).
d_14 = ([
    'demos-2019-08-23-11-29-pol-random-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-23-11-28-pol-random-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-23-11-28-pol-random-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-23-11-28-pol-random-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-23-11-28-pol-random-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    ],
    'demos-2019-08-23-pol-random-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl',
)

# Tier 3, random policy (hermes1).
d_15 = ([
    'demos-2019-08-22-19-50-pol-random-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-50-pol-random-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-50-pol-random-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-50-pol-random-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    ],
    'demos-2019-08-22-pol-random-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl',
)


# ------------------------------------------------------------------------------------------------ #
# HIGHEST POINT POLICY BASELINE (with heavier domain randomization 2.0 but that's not as relevant)
# ------------------------------------------------------------------------------------------------ #

# Tier 1, highest point (triton4)
d_16 = ([
    'demos-2019-08-22-17-12-pol-highest-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-17-12-pol-highest-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-17-12-pol-highest-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-17-12-pol-highest-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-22-17-12-pol-highest-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    ],
    'demos-2019-08-22-pol-highest-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl',
)

# Tier 2, highest point (triton4 and hermes1 together)
d_17 = ([
    'demos-2019-08-23-16-16-pol-highest-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_333.pkl',
    'demos-2019-08-23-16-15-pol-highest-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_333.pkl',
    'demos-2019-08-23-16-15-pol-highest-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_333.pkl',
    'demos-2019-08-23-16-15-pol-highest-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_333.pkl',
    'demos-2019-08-23-16-15-pol-highest-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_333.pkl',
    'demos-2019-08-23-19-03-pol-highest-seed-1342-clip_a-True-delta_a-True-obs-blender-tier2_epis_167.pkl',
    'demos-2019-08-23-19-03-pol-highest-seed-1343-clip_a-True-delta_a-True-obs-blender-tier2_epis_168.pkl',
    ],
    'demos-2019-08-23-pol-highest-seed-1337_to_1343-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl',
)

# Tier 3, highest point (hermes1)
d_18 = ([
    'demos-2019-08-22-19-52-pol-highest-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-52-pol-highest-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-52-pol-highest-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    'demos-2019-08-22-19-52-pol-highest-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
    ],
    'demos-2019-08-22-pol-highest-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl',
)


# ------------------------------------------------------------------------------------------------ #
# FROM ADI, oracle, corner REVEALER policy.
# ------------------------------------------------------------------------------------------------ #

## Tier 1, revealer, hermes1, from me, with exact corner (not 'one inwards')
#d_30_tier1_old = ([
#    'demos-2019-08-27-19-18-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-18-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-18-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-18-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-19-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-20-pol-oracle_reveal-seed-1342-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-20-pol-oracle_reveal-seed-1343-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    'demos-2019-08-27-19-21-pol-oracle_reveal-seed-1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
#    ],
#    'demos-2019-08-27-pol-oracle_reveal-seed-1337_to_1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl',
#)

# Tier 1, revealer, Triton1, from me, WITH CORNER INWARDS (so try this for final evaluation).
d_30_tier1 = ([
    'demos-2019-08-28-14-08-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-09-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-09-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-09-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-10-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-10-pol-oracle_reveal-seed-1342-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-10-pol-oracle_reveal-seed-1343-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    'demos-2019-08-28-14-10-pol-oracle_reveal-seed-1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_250.pkl',
    ],
    'demos-2019-08-28-pol-oracle_reveal-seed-1337_to_1344-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl',
)

# Tier 2, revealer, Triton4, from me, WITH CORNER INWARDS (so try this for final evaluation).
# EDIT: argh this froze ...
#d_30_tier2 = ([
#    'demos-2019-08-28-14-23-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
#    'demos-2019-08-28-14-23-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
#    'demos-2019-08-28-14-23-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
#    'demos-2019-08-28-14-23-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
#    'demos-2019-08-28-14-23-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
#    ],
#    'demos-2019-08-28-pol-oracle_reveal-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl',
#)

d_30_tier2 = ([
    '',
    '',
    '',
    '',
    '',
    ],
    '',
)

## Tier 3, highest point (triton4) NOTE: from Adi, with the exact corner (not 'one inwards')
#d_30_tier3_old = ([
#    'demos-2019-08-25-22-50-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
#    'demos-2019-08-25-22-51-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
#    'demos-2019-08-25-22-51-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
#    'demos-2019-08-25-22-52-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_500.pkl',
#    ],
#    'demos-2019-08-25-pol-oracle_reveal-seed-1337_to_1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl',
#)

# Tier 3, revealer, Triton4, from me, WITH CORNER INWARDS (so try this for final evaluation).
# EDIT: argh this froze ...
#d_30_tier3 = ([
#    'demos-2019-08-28-14-20-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
#    'demos-2019-08-28-14-20-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
#    'demos-2019-08-28-14-20-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
#    'demos-2019-08-28-14-20-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
#    'demos-2019-08-28-14-20-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
#    ],
#    'demos-2019-08-28-pol-oracle_reveal-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_200_COMBINED.pkl',
#)

# Tier 3, revealer, Triton4, from me, WITH CORNER INWARDS (so try this for final evaluation).
d_30_tier3 = ([
    'demos-2019-08-29-19-21-pol-oracle_reveal-seed-1337-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-29-19-22-pol-oracle_reveal-seed-1338-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-29-19-22-pol-oracle_reveal-seed-1339-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-29-19-22-pol-oracle_reveal-seed-1340-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-29-19-22-pol-oracle_reveal-seed-1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    ],
    'demos-2019-08-29-pol-oracle_reveal-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_2000_COMBINED.pkl',
)








## USE THESE!

# ------------------------------------------------------------------------------------------------ #
# DEPTH IMAGES !!!! ORACLE CORNER POLICY, not the revealer ...
# heavier domain randomization 2.0 (with depth images of course)
# ------------------------------------------------------------------------------------------------ #

# Tier 1 (Triton1)
d_40_tier1_depth = ([
    'demos-2019-08-28-22-14-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-28-22-14-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-28-22-14-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-28-22-14-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    'demos-2019-08-28-22-14-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_400.pkl',
    ],
    'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier1_epis_2000_COMBINED.pkl',
)

# Tier 2 (hermes1)
d_40_tier2_depth = ([
    'demos-2019-08-28-22-02-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-28-22-02-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-28-22-02-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-28-22-03-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    'demos-2019-08-28-22-03-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_400.pkl',
    ],
    'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier2_epis_2000_COMBINED.pkl',
)

# Tier 3 (Triton1)
d_40_tier3_depth = ([
    'demos-2019-08-28-22-17-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-28-22-17-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-28-22-17-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-28-22-17-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    'demos-2019-08-28-22-17-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_400.pkl',
    ],
    'demos-2019-08-28-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-tier3_epis_2000_COMBINED.pkl',
)

# ------------------------------------------------------------------------------------------------ #
# COLOR IMAGES !!!! ORACLE CORNER POLICY, not the revealer ...
# heavier domain randomization 2.0 (with RGB images of course, and the noise injection afterwards)
# ------------------------------------------------------------------------------------------------ #

# Tier 1 (Triton1, aug 30)
d_40_tier1_color = ([
    'demos-2019-08-30-13-13-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_400.pkl',
    'demos-2019-08-30-13-14-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_400.pkl',
    'demos-2019-08-30-13-14-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_400.pkl',
    'demos-2019-08-30-13-14-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_400.pkl',
    'demos-2019-08-30-13-14-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_400.pkl',
    ],
    'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier1_epis_2000_COMBINED.pkl',
)

# Tier 2 (hermes1, aug 30)
d_40_tier2_color = ([
    'demos-2019-08-30-13-22-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_400.pkl',
    'demos-2019-08-30-13-22-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_400.pkl',
    'demos-2019-08-30-13-22-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_400.pkl',
    'demos-2019-08-30-13-22-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_400.pkl',
    'demos-2019-08-30-13-22-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_400.pkl',
    ],
    'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier2_epis_2000_COMBINED.pkl',
)

# Tier 3 (Triton1, aug 30)
d_40_tier3_color = ([
    'demos-2019-08-30-21-20-pol-oracle-seed-1337-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-30-21-20-pol-oracle-seed-1338-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-30-21-20-pol-oracle-seed-1339-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-30-21-20-pol-oracle-seed-1340-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    'demos-2019-08-30-21-20-pol-oracle-seed-1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_400.pkl',
    ],
    'demos-2019-08-30-pol-oracle-seed-1337_to_1341-clip_a-True-delta_a-True-obs-blender-depthimg-False-tier3_epis_2000_COMBINED.pkl',
)


# ------------------------------------------------------------------------------------------------ #
# WRINKLES
# ------------------------------------------------------------------------------------------------ #

d_wrinkle_t1 = ([
    'demos-2019-09-01-12-10-pol-wrinkle-seed-1000-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_400.pkl',
    'demos-2019-09-01-12-10-pol-wrinkle-seed-1001-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_400.pkl',
    'demos-2019-09-01-12-12-pol-wrinkle-seed-1002-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_400.pkl',
    'demos-2019-09-01-12-12-pol-wrinkle-seed-1003-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_400.pkl',
    'demos-2019-09-01-12-13-pol-wrinkle-seed-1004-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_400.pkl',
    ],
    'demos-2019-09-01-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier1_epis_2000_COMBINED.pkl',
)

d_wrinkle_t2 = ([
    'demos-2019-09-02-08-15-pol-wrinkle-seed-1000-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_400.pkl',
    'demos-2019-09-02-08-15-pol-wrinkle-seed-1001-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_400.pkl',
    'demos-2019-09-02-08-16-pol-wrinkle-seed-1002-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_400.pkl',
    'demos-2019-09-02-08-16-pol-wrinkle-seed-1003-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_400.pkl',
    'demos-2019-09-02-08-16-pol-wrinkle-seed-1004-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_400.pkl',
    ],
    'demos-2019-09-02-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier2_epis_2000_COMBINED.pkl',
)

d_wrinkle_t3 = ([
    'demos-2019-09-03-08-17-pol-wrinkle-seed-1000-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_400.pkl',
    'demos-2019-09-03-08-17-pol-wrinkle-seed-1001-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_400.pkl',
    'demos-2019-09-03-08-18-pol-wrinkle-seed-1002-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_400.pkl',
    'demos-2019-09-03-08-18-pol-wrinkle-seed-1003-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_400.pkl',
    'demos-2019-09-03-08-18-pol-wrinkle-seed-1004-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_400.pkl',
    ],
    'demos-2019-09-03-pol-wrinkle-seed-1000_to_1004-clip_a-True-delta_a-True-obs-1d-img-False-tier3_epis_2000_COMBINED.pkl',
)

# ------------------------------------------------------------------------------------------------ #
# RGBD
# ------------------------------------------------------------------------------------------------ #

d_rgbd_t1 = ([
    'demos-2020-02-09-16-31-pol-oracle-seed-1336-obs-blender-depth-False-rgbd-True-tier1_epis_400.pkl',
    'demos-2020-02-09-16-31-pol-oracle-seed-1337-obs-blender-depth-False-rgbd-True-tier1_epis_400.pkl',
    'demos-2020-02-09-16-31-pol-oracle-seed-1338-obs-blender-depth-False-rgbd-True-tier1_epis_400.pkl',
    'demos-2020-02-09-16-31-pol-oracle-seed-1339-obs-blender-depth-False-rgbd-True-tier1_epis_400.pkl',
    'demos-2020-02-09-16-31-pol-oracle-seed-1340-obs-blender-depth-False-rgbd-True-tier1_epis_400.pkl',
    ],
    'demos-2020-02-09-16-31-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier1_epis_2000_COMBO.pkl'
)

d_rgbd_t2 = ([
    'demos-2020-02-10-15-00-pol-oracle-seed-1336-obs-blender-depth-False-rgbd-True-tier2_epis_400.pkl',
    'demos-2020-02-10-15-01-pol-oracle-seed-1337-obs-blender-depth-False-rgbd-True-tier2_epis_400.pkl',
    'demos-2020-02-10-15-01-pol-oracle-seed-1338-obs-blender-depth-False-rgbd-True-tier2_epis_400.pkl',
    'demos-2020-02-10-15-02-pol-oracle-seed-1339-obs-blender-depth-False-rgbd-True-tier2_epis_400.pkl',
    'demos-2020-02-10-15-02-pol-oracle-seed-1340-obs-blender-depth-False-rgbd-True-tier2_epis_400.pkl',
    ],
    'demos-2020-02-10-15-02-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier2_epis_2000_COMBO.pkl'
)

d_rgbd_t3 = ([
    'demos-2020-02-10-15-04-pol-oracle-seed-1336-obs-blender-depth-False-rgbd-True-tier3_epis_400.pkl',
    'demos-2020-02-10-15-04-pol-oracle-seed-1337-obs-blender-depth-False-rgbd-True-tier3_epis_400.pkl',
    'demos-2020-02-10-15-04-pol-oracle-seed-1338-obs-blender-depth-False-rgbd-True-tier3_epis_400.pkl',
    'demos-2020-02-10-15-04-pol-oracle-seed-1339-obs-blender-depth-False-rgbd-True-tier3_epis_400.pkl',
    'demos-2020-02-10-15-05-pol-oracle-seed-1340-obs-blender-depth-False-rgbd-True-tier3_epis_400.pkl',
    ],
    'demos-2020-02-10-15-05-pol-oracle-seed-1336_to_1340-obs-blender-depth-False-rgbd-True-tier3_epis_2000_COMBO.pkl'
)


# ------------------------------------ ADJUST ------------------------------------ #
dataset = d_rgbd_t3
# ------------------------------------ ADJUST ------------------------------------ #


pickle_files = dataset[0]
target_file = join('logs',dataset[1])

combo = []
for pf in pickle_files:
    with open(join('logs',pf), 'rb') as fh:
        data = pickle.load(fh)
    print('  just loaded {}, len {}'.format(pf, len(data)))
    combo.extend(data)

print('saving at {}, len {}'.format(target_file, len(combo)))
with open(target_file, 'wb') as fh:
    data = pickle.dump(combo, fh)
