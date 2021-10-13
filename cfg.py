"""Important vars that need to be globally accessible across modules."""

WORK_DIR = None
SAVE_DIR = None
SNAP_DIR = None

WORLD_CONFIG = {}

# KD Tree
KDT = None

# For image exports, each pixel's 2D latitude/longitude is converted to 3D XYZ
# and then a query is run against the KD Tree to find neighbors and distances.
IMG_QUERY_DATA = []
