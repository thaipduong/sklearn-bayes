import numpy as np

nll = 0.0
for i in range(4000):
	nll = nll - np.log(0.5)
nll = nll/4000
print("NLL = ", nll)