import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

sampling_check = [[0.581995, 1.11355, 2.76265, 5.38922, 10.8638, 16.0433, 20.8066, 25.829, 31.3338, 35.6349, 39.1454, 44.5611, 48.7449 ], \
                  [0.290137, 0.591931, 1.36886, 2.72081, 5.4052, 7.96275, 10.5356, 13.1085, 15.3395, 17.3946, 19.9424, 21.8291, 23.9875], \
                  [0.203377, 0.38169, 0.927818, 1.82397, 3.55651, 5.32779, 6.99395, 8.68606, 10.1679, 11.6599, 13.2711, 14.8899, 16.2133 ], \
                  [0.1615, 0.286974, 0.741545, 1.37135, 2.88115, 4.00612, 5.26615, 6.54635, 7.74868, 8.88332, 10.0699, 11.154, 12.317 ], \
                  [0.130631, 0.24078, 0.570535, 1.09401, 2.22354, 3.22366, 4.2474, 5.22664, 6.39662, 7.33635, 8.37065, 9.02519, 9.8622],\
                  [0.0960018, 0.164917, 0.390812, 0.833468, 1.44482, 2.2349, 2.89098, 3.55507, 4.18217, 4.84471, 5.46047, 6.01834, 6.60237 ], \
                  [0.0748277, 0.128033, 0.298122, 0.557382, 1.11695, 1.65736, 2.20315, 2.62596, 3.1843, 3.64393, 4.09514, 4.57259, 5.00604], \
                  [0.0298844, 0.0407432, 0.0749596, 0.128183, 0.23883, 0.345288, 0.447952, 0.538716, 0.640913, 0.765968, 0.845751, 0.921422, 0.995222 ], \
                  [0.0245127, 0.029936, 0.0461679, 0.0779107, 0.133236, 0.179266, 0.2524, 0.361053, 0.355877, 0.384636, 0.43089, 0.500038, 0.517807 ]]
our_check =  [5.21855, 5.2337, 5.26961, 5.38622, 5.37006, 5.51531, 5.45334, 5.54975, 5.53303, 5.59455, 5.58507, 5.66692, 5.67555 ]


f, ax = plt.subplots(figsize=(11,5))
length = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
width = 2
#ax.plot(length, sampling_check[0], 'g--', marker='>', label='$\Delta$ = 0.001m')
ax.plot(length, sampling_check[1], 'g--', marker='^', linewidth=width,markersize=9,label='$\Delta$ = 0.002m')
ax.plot(length, sampling_check[2], 'g--', marker='>', linewidth=width,markersize=9,label='$\Delta$ = 0.003m')
#ax.plot(length, sampling_check[3], 'g--', marker='o',label='$\Delta$ = 0.004m')
ax.plot(length, sampling_check[4], 'g--', marker='x', linewidth=width,markersize=9,label='$\Delta$ = 0.005m')
ax.plot(length, sampling_check[5], 'g--', marker='v', linewidth=width,markersize=9,label='$\Delta$ = 0.01m')
ax.plot(length, sampling_check[6], 'g--', marker='<', linewidth=width,markersize=9,label='$\Delta$ = 0.05m')
ax.plot(length, sampling_check[7], 'g--', marker='o', linewidth=width,markersize=9,label='$\Delta$ = 0.1m')
ax.plot(length, our_check, 'r', marker='o', linewidth=width,markersize=9,label='Ours')
ax.set_xlabel("Length(m)", size='x-large')
ax.set_ylabel("time($\mu$s)", size='x-large')
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.legend(facecolor='xkcd:silver',fontsize='x-large')
plt.savefig("/home/erl/repos/sklearn-bayes/data/results/compare_sampling.pdf",bbox_inches='tight', pad_inches = 0)
plt.show()