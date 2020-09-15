import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
plt.rcParams['text.usetex'] = True
sampling_check = [[0.562083, 1.10585, 2.71444, 5.42414, 10.7492, 15.9105, 20.9284, 25.3914, 30.0854, 34.7675, 38.5098, 43.1469, 47.2701 ], \
                  [0.289906, 0.556072, 1.35131, 2.66474, 5.24553, 7.77971, 10.2637, 12.6399, 15.0156, 17.2788, 19.4786, 21.6199, 23.6393], \
                  [0.205339, 0.379962, 0.908342, 1.79011, 3.51452, 5.19741, 6.86078, 8.46691, 10.0223, 11.5354, 13.0106, 14.4254, 15.7804 ], \
                  [0.15762, 0.290408, 0.690921, 1.34618, 2.64603, 3.9027, 5.24372, 6.44479, 7.52794, 8.64482, 9.75768, 10.7918, 11.8178 ], \
                  [0.129029, 0.23656, 0.553472, 1.07916, 2.11806, 3.14103, 4.14053, 5.10959, 6.08368, 6.99469, 7.83168, 8.66037, 9.55741],\
                  [0.0993544, 0.167994, 0.38192, 0.736179, 1.42685, 2.10359, 2.77263, 3.41644, 4.04254, 4.66322, 5.26712, 5.80751, 6.37498 ], \
                  [0.0775153, 0.130882, 0.291974, 0.556024, 1.07584, 1.59517, 2.0909, 2.57691, 3.0525, 3.4796, 3.92349, 4.37085, 4.79562], \
                  [0.0318978, 0.0430341, 0.078143, 0.132992, 0.237528, 0.340729, 0.44293, 0.540934, 0.638937, 0.731114, 0.824133, 0.912991, 0.996608 ], \
                  [0.0260762, 0.0324303, 0.0498636, 0.0781447, 0.132194, 0.183616, 0.235186, 0.284312, 0.333965, 0.380986, 0.426552, 0.471901, 0.512791]]
#K = 10 our_check =  [5.06892, 5.10366, 5.16264, 5.29786, 5.45316, 5.39715, 5.54821, 5.5322, 5.44904, 5.47472, 5.64255, 5.68957, 5.65687]
our_check =  [2.72985, 2.73979, 2.77868, 2.81123, 2.85098, 2.87696, 2.89393, 2.91202, 2.92018, 2.9348, 2.95863, 2.95989, 2.96398]

f, ax = plt.subplots(figsize=(5.625,4.5))
length = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
res = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01, 0.05, 0.1]
width = 2
marksize = 6
#ax.plot(length, sampling_check[0], 'g--', marker='>', label='$\Delta$ = 0.001m')
ax.plot(length, sampling_check[1], 'g--', marker='^', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.002m')
ax.plot(length, sampling_check[2], 'g--', marker='>', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.003m')
#ax.plot(length, sampling_check[3], 'g--', marker='o',label='$\Delta$ = 0.004m')
ax.plot(length, sampling_check[4], 'g--', marker='x', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.005m')
ax.plot(length, sampling_check[6], 'g--', marker='v', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.01m')
ax.plot(length, sampling_check[7], 'g--', marker='<', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.05m')
ax.plot(length, sampling_check[8], 'g--', marker='o', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.1m')
ax.plot(length, our_check, 'r', marker='o', linewidth=width,markersize=marksize,label='Ours')
ax.set_xlabel("Trajectory time length $t_f$", size='x-large')
ax.set_ylabel("time($\mu$s)", size='x-large')
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.legend(facecolor='xkcd:silver',fontsize='x-large')
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/compare_sampling.eps",bbox_inches='tight', pad_inches = 0)
plt.show()


sampling_check = [[1.67783, 3.31686, 8.42008, 16.5443, 32.8667, 49.2169, 66.0718, 82.9295, 99.2981, 117.335, 133.256, 146.823, 160.299 ], \
                  [0.827488, 1.6497, 4.13882, 8.23888, 16.4386, 24.6858, 32.8261, 41.0405, 49.4816, 57.48, 65.3256, 72.9202, 80.496], \
                  [0.575389, 1.1159, 2.75358, 5.51002, 10.9839, 16.4785, 22.0621, 27.6035, 32.9691, 38.4712, 43.8354, 48.7673, 53.6245 ], \
                  [0.418679, 0.835855, 2.07337, 4.14478, 8.30439, 12.3169, 16.3721, 20.6581, 24.7698, 28.9248, 32.8743, 36.6143, 40.5832 ], \
                  [0.336978, 0.654982, 1.61856, 3.22524, 6.43993, 9.64929, 12.8796, 16.109, 19.3275, 22.5472, 25.5827, 28.5544, 31.4268],\
                  [0.238986, 0.447332, 1.09512, 2.16813, 4.30693, 6.44674, 8.61892, 10.7469, 12.9457, 15.0839, 17.1372, 19.1428, 21.101], \
                  [0.177423, 0.34237, 0.831591, 1.64953, 3.26748, 4.87112, 6.46268, 8.07565, 9.69807, 11.308, 12.8683, 14.3849, 15.8922], \
                  [0.0378165, 0.0710979, 0.17394, 0.336524, 0.660688, 1.00759, 1.31735, 1.63843, 1.9688, 2.31988, 2.61988, 2.94761, 3.22138], \
                  [0.0196378, 0.0380577, 0.0873967, 0.176275, 0.341881, 0.508528, 0.673901, 0.838112, 1.00323, 1.17084, 1.3296, 1.48507, 1.66175 ]]
#our_check =  [3.68586, 3.8759, 4.48159, 5.63, 8.06262, 10.8922, 13.3258, 15.6952, 17.5617, 18.421, 18.8549, 18.9216, 18.9891]
# K = 10  our_check =  [6.18544, 6.50756, 7.60805, 9.60893, 13.9589, 18.5172, 22.2865, 26.1083, 29.1015, 30.603, 31.1681, 31.6523, 31.8443 ]
our_check = [2.38526, 2.46122, 2.88015, 3.63324, 5.17813, 6.86875, 8.19975, 9.57318, 10.5106, 11.043, 11.241, 11.3572, 11.4113]

f, ax = plt.subplots(figsize=(5.625,4.5))
length = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
width = 2
#ax.plot(length, sampling_check[0], 'g--', marker='>', label='$\Delta$ = 0.001m')
ax.plot(length, sampling_check[1], 'g--', marker='^', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.002m')
ax.plot(length, sampling_check[2], 'g--', marker='>', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.003m')
#ax.plot(length, sampling_check[3], 'g--', marker='o',label='$\Delta$ = 0.004m')
ax.plot(length, sampling_check[4], 'g--', marker='x', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.005m')
ax.plot(length, sampling_check[6], 'g--', marker='v', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.01m')
ax.plot(length, sampling_check[7], 'g--', marker='<', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.05m')
ax.plot(length, sampling_check[8], 'g--', marker='o', linewidth=width,markersize=marksize,label='$SB, \Delta$ = 0.1m')
ax.plot(length, our_check, 'r', marker='o', linewidth=width,markersize=marksize,label='Ours')
ax.set_xlabel("Trajectory time length $t_f$", size='x-large')
ax.set_ylabel("time($\mu$s)", size='x-large')
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.legend(facecolor='xkcd:silver',fontsize='x-large')
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/compare_sampling_fas.eps",bbox_inches='tight', pad_inches = 0)
plt.show()