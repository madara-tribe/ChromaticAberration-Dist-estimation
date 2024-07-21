import numpy as np

f = 24 #[mm]
a = 18
vf = 35
p = 1
# b = (a*vf/2p) (1/f - 1/vf - 1/x)

def formula_bsize(x):
    S = (vf*a)/(2*p)
    b = S*(1/f - 1/vf - np.round(1/x, 4))
    print("b", b)
