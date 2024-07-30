"""
Convert the MATLAB Function to a Python Function:
MATLAB uses function to define functions, whereas Python uses the def keyword.
In MATLAB, 2*[1:12] creates an array from 2 to 24 with a step of 2. In Python, you can achieve this using numpy.arange() or list comprehensions.
Using NumPy for Array Creation:
MATLAB arrays are 1-based indexed, while Python arrays are 0-based indexed. However, this indexing difference doesn't affect the generation of the array.
"""

import numpy as np

def candidate_depths():
    # Generate an array similar to 2*[1:12] in MATLAB
    depths = 2 * np.arange(1, 13)
    return depths

# Example usage
depths = candidate_depths()
print(depths)
