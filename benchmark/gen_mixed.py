import numpy as np
rng = np.random.default_rng(42)
d = np.concatenate([rng.normal(-5,1,500), rng.gamma(4,2,500)])
np.savetxt('/tmp/mixed_data.txt', d, fmt='%.6f')
print('Saved', len(d), 'values: 500 from N(-5,1) + 500 from Gamma(4,2)')
