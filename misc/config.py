# Metadata.
global AUTHOR
global LICENCE
global EMAIL
global STATUS
global STATUS
global DOCFORMAT

AUTHOR = "Leo Tisljaric"
LICENCE = "GNU General Public License v3.0"
EMAIL = "tisljaricleo@gmail.com"
STATUS = "Development"
DOCFORMAT = "reStructuredText"

# STM setup.
global RESOLUTION
global MAX_INDEX
global MAX_ITER
global SPEED_LIST
global SPEED_TYPE
global SPEED_LIMIT_TRESH
global SL_DOWN
global SL_UP
global DIAG_LOCS

RESOLUTION = int(5)  # Resolution of the speed transition matrix in km/h
MAX_INDEX = int(100 / RESOLUTION)  # Maximum index of the numpy array.
MAX_ITER = int(100 + RESOLUTION)  # Maximal iteration for the range() function.
SPEED_LIST = list(range(RESOLUTION, MAX_ITER, RESOLUTION))  # All speed values for rows/columns of the matrix.
SPEED_TYPE = 'rel'
SL_DOWN = 50
SL_UP = 80
SPEED_LIMIT_TRESH = 50

diag_locs = []
for i in range(0, MAX_INDEX):
    for j in range(0, MAX_INDEX):
        if i == j:
            diag_locs.append((i, j))
DIAG_LOCS = diag_locs
