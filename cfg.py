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
# Item 0 is a ndarray of the distances to the 3 nearest neighbors
#     with dtype np.float64 if the image is small, else np.float32.
# Item 1 is a ndarray of the indices of the 3 nearest neighbors
#     with dtype np.int64 if the image is small, else np.uint32
IMG_QUERY_DATA = (None, None)
