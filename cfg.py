"""Important vars that need to be globally accessible across modules."""

WORK_DIR = None
SAVE_DIR = None
SNAP_DIR = None

WORLD_CONFIG = {}

# KD Tree
KDT = None

# For image exports, each pixel's 2D latitude/longitude is converted to 3D XYZ
# and then a query is run against the KD Tree to find neighbors and distances.
# Results are stored as a tuple of 2 ndarrays, both with shape (height, width, 3).
# Item 0 is a ndarray of the distances to the 3 nearest neighbors with dtype np.float64.
# Item 1 is a ndarray of the indices of the 3 nearest neighbors with dtype np.int64.
# NOTE: For particularly large image export resolutions we might consider converting
# these to float32 and int32 (or uint32) if the size in RAM becomes a problem.
IMG_QUERY_DATA = ()
