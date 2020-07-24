import numpy as np

def remove_interior(ground_truth):
    ground_truth[5:8, 59] = 0
    ground_truth[5:8, 63] = 0

    ground_truth[29:32, 59] = 0
    ground_truth[29:32, 63] = 0

    ground_truth[48:51, 82] = 0
    ground_truth[48:51, 86] = 0
    ground_truth[48:51, 89] = 0
    ground_truth[48:51, 93] = 0

    ground_truth[33:38, 64] = 0
    ground_truth[33:38, 84] = 0
    ground_truth[33:38, 92] = 0
    ground_truth[33:38, 112] = 0

    ground_truth[85, 82:85] = 0
    ground_truth[81, 82:85] = 0

    ground_truth[74, 104:107] = 0
    ground_truth[70, 104:107] = 0

    ground_truth[28, 107:117] = 0
    ground_truth[25:32, 110] = 0
    ground_truth[25:32, 113] = 0

    ground_truth[61:66, 64] = 0
    ground_truth[61:66, 84] = 0
    ground_truth[61:66, 92] = 0
    ground_truth[61:66, 112] = 0

    ground_truth[8, 107:117] = 0
    ground_truth[5:12, 110] = 0
    ground_truth[5:12, 113] = 0

    ground_truth[85, 126:133] = 0
    ground_truth[81, 126:133] = 0
    ground_truth[78:89, 129] = 0

    ground_truth[85, 229:236] = 0
    ground_truth[81, 229:236] = 0
    ground_truth[78:89, 232] = 0

    ground_truth[85, 185:188] = 0
    ground_truth[81, 185:188] = 0

    ground_truth[74, 152:159] = 0
    ground_truth[70, 152:159] = 0
    ground_truth[67:78, 155] = 0

    ground_truth[74, 207:210] = 0
    ground_truth[70, 207:210] = 0

    ground_truth[74, 255:261] = 0
    ground_truth[70, 255:261] = 0
    ground_truth[67:78, 258] = 0

    ground_truth[48:51, 144] = 0
    ground_truth[48:51, 148] = 0
    ground_truth[48:51, 151] = 0
    ground_truth[48:51, 155] = 0

    ground_truth[28, 188:198] = 0
    ground_truth[25:32, 191] = 0
    ground_truth[25:32, 195] = 0

    ground_truth[8, 188:198] = 0
    ground_truth[5:12, 191] = 0
    ground_truth[5:12, 195] = 0

    ground_truth[48:51, 228] = 0
    ground_truth[48:51, 232] = 0
    ground_truth[48:51, 236] = 0
    ground_truth[48:51, 239] = 0

    ground_truth[61:66, 149] = 0
    ground_truth[61:66, 168] = 0

    ground_truth[61:66, 233] = 0
    ground_truth[61:66, 253] = 0

    ground_truth[33:38, 233] = 0
    ground_truth[33:38, 253] = 0

    ground_truth[33:38, 149] = 0
    ground_truth[33:38, 168] = 0
    ground_truth[33:38, 177] = 0
    ground_truth[33:38, 196] = 0

    ground_truth[4, 57:67] = 0
    ground_truth[4, 107:117] = 0
    ground_truth[4, 188:198] = 0
    ground_truth[4, 226:231] = 0

    ground_truth[4, 67] = 1
    ground_truth[4, 117] = 1
    ground_truth[4, 198] = 1
    ground_truth[4, 231] = 1

    ground_truth[32, 57:67] = 0
    ground_truth[32, 107:117] = 0
    ground_truth[32, 188:198] = 0
    ground_truth[66, 152:159] = 0
    ground_truth[66, 255:261] = 0
    ground_truth[66, 104:107] = 0
    ground_truth[89, 229:233] = 0

    ground_truth[90:95, 56] = 0
    ground_truth[106:114, 56] = 0






def get_goundtruth_map():
	path = "/home/erl/repos/sklearn-bayes/data/groundtruth/"
	A = np.loadtxt(path + 'fla_warehouse1.cfg')
	A = A.reshape(343, 127, 27)
	groundtruth_map = A[:,:,1]
	groundtruth_map = groundtruth_map.transpose()
	remove_interior(groundtruth_map)
	return groundtruth_map


def compare(map1, map2, drift_allowance = 0):
    error = 0
    for i in range(map1.shape[0]):
        for j in range(map1.shape[1]):
            correct = False
            if map1[i,j] == map2[i, j]:
                correct = True
            else:
                count = 0
                for k in range(-drift_allowance, drift_allowance+1):
                    for l in range(-drift_allowance, drift_allowance+1):
                        if 0<= i + k < map1.shape[0] and 0<= j + l < map1.shape[1]:
                            if map1[i,j] == map2[i+k, j+l]:
                                count = count + 1
                if 0< count:
                    correct = True
            if not correct:
                error = error + 1
    return error

def compare_tpr(map1, map2, drift_allowance = 0):
    error = 0
    count_true = 0
    for i in range(map1.shape[0]):
        for j in range(map1.shape[1]):
            if map1[i, j] < 0.5:
                continue
            count_true = count_true + 1
            correct = False
            if map1[i,j] == map2[i, j]:
                correct = True
            else:
                count = 0
                for k in range(-drift_allowance, drift_allowance+1):
                    for l in range(-drift_allowance, drift_allowance+1):
                        if 0<= i + k < map1.shape[0] and 0<= j + l < map1.shape[1]:
                            if map1[i,j] == map2[i+k, j+l]:
                                count = count + 1
                if 0< count:
                    correct = True
            if not correct:
                error = error + 1
    #print("count_true ", count_true)
    return error, count_true