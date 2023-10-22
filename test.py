import numpy as np

dist_to_marker = 0.5
dist_update = np.maximum(dist_to_marker - 0.15, 0)
print(dist_update)