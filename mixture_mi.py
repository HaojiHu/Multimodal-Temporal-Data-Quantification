import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict,  Counter
from scipy.special import psi
from entropy_mao import entropy_estimate_1d_continuous, entropy_estimate_1d_discrete
import mixture_dataset as md
import math



def obtain_repeated_values(time_series):
    val2freq = defaultdict(int)
    for val in time_series:
        val2freq[val] += 1

    for val in list(val2freq.keys()):
        if val2freq[val] == 1:
            del val2freq[val]

    return val2freq


def partition_data_pairs(disc, cont):
    val2freq = obtain_repeated_values(cont)
    dis_dis = []
    dis_cont = []
    for i, val in enumerate(cont):
        if val in val2freq:
            dis_dis.append([disc[i], val])
        else:
            dis_cont.append([disc[i], val])



    # value_freq = Counter(val2freq.values())
    # print(value_freq)


    dis_dis = tuple(np.asarray(row) for row in zip(*dis_dis))
    dis_cont = tuple(np.asarray(row) for row in zip(*dis_cont))



    return dis_dis, dis_cont, len(val2freq)

def mixture_mi(disc, cont):
    dis_dis, dis_cont, distinct_repeated_values = partition_data_pairs(disc, cont)
    if len(dis_dis) == 0:
        H_Y_A = 0
    else:
        H_Y_A = entropy_estimate_1d_discrete(dis_dis[1])
    # print(f"H_Y_A: {H_Y_A} \n")

    H_XY_A = 0
    groups = dict()
    if len(dis_dis) > 0:
        for x, y in zip(dis_dis[0], dis_dis[1]):
            if (x, y) in groups:
                groups[(x, y)] += 1
            else:
                groups[(x, y)] = 1

        for k in groups.keys():
            p = groups[k] / len(dis_dis[0])
            H_XY_A += -p * np.log(p)
    # print(f"H_XY_A: {H_XY_A}\n")

    if len(dis_cont) > 1 and len(dis_cont[1]) > 1:
        H_Y_B = entropy_estimate_1d_continuous(dis_cont[1])
    else:
        H_Y_B = 0
    # print(f"H_Y_B: {H_Y_B}\n")
    if len(dis_dis) == 0:
        p_a = 0
    else:
        p_a = len(dis_dis[0]) / len(disc)

    H_XY_B = 0
    if len(dis_cont) > 1:
        discrete_range = set(dis_cont[0])
        # print(f"discrete_range: {discrete_range}")
        groups = defaultdict(np.array)
        for d in discrete_range:
            groups[d] = dis_cont[1][dis_cont[0] == d]
        for g in groups.keys():
            if len(groups[g]) > 1:
                H_XY_B += len(groups[g]) / (len(dis_cont[0])) * entropy_estimate_1d_continuous(groups[g], islog2=False)
                H_XY_B -= len(groups[g]) / (len(dis_cont[0]))  * np.log(len(groups[g]) / (len(dis_cont[0])) )
    else:
        H_XY_B = 0
    # print(f"H_XY_B: {H_XY_B}\n")

    H_X = entropy_estimate_1d_discrete(disc)
    # print(f"H_X: {H_X}\n")

    # p_b = len(dis_cont[0]) / len(disc)
    p_b = 1 - p_a
    # print(f"p_a: {p_a}")
    # print(f"p_b: {p_b}")
    MI = p_a * (H_Y_A - H_XY_A) + p_b * (H_Y_B - H_XY_B) + H_X
    # print("p_a * (H_Y_A - H_XY_A) : ", p_a * (H_Y_A - H_XY_A) )
    # print(f"MI: {MI}")

    return max(MI, 0)

def NMI(discs, conts, isNormalized=True):

    scale = np.max(conts) - np.min(conts)
    conts = (conts - np.min(conts)) / scale
    # disc = entropy_estimate_1d_discrete(discs, islog2=True)


    unit = 1
    for i in range(len(conts)-1):
        if unit > conts[i] - np.min(conts) > 0:
            unit = conts[i] - np.min(conts)
    unit = unit / len(conts)

    # cont = entropy_estimate_1d_continuous(conts, islog2=True, unit=unit) + np.log2(scale)

    # print('scale: ', np.log(scale))
    mi = mixture_mi(discs, conts)
    # print("un-normalized mi in mixture_mi.py: ", mi)

    # return 2 * mi / (disc + cont)
    # return mi / min(disc, cont)
    # return mi / cont
    if not isNormalized:
        return mi
    return math.sqrt(1 - math.exp((-2*mi)/(unit+1)))

if __name__ == "__main__":

    arrs = md.get_mixture(1000)
    dis_dis, dis_cont = md.unmix(arrs)

    print(dis_dis)
    print(dis_cont)

    H_Y_A = entropy_estimate_1d_discrete(dis_dis[1])
    print(f"H_Y_A: {H_Y_A} \n")

    H_XY_A = 0
    groups = dict()
    for x, y in zip(dis_dis[0], dis_dis[1]):
        if (x,y) in groups:
            groups[(x,y)] += 1
        else:
            groups[(x,y)] = 1

    for k in groups.keys():
        p = groups[k]/len(dis_dis[0])
        H_XY_A += -p * np.log(p)
    print(f"H_XY_A: {H_XY_A}\n")

    H_Y_B = entropy_estimate_1d_continuous(dis_cont[1])
    print(f"H_Y_B: {H_Y_B}\n")


    H_XY_B = 0
    discrete_range = set(dis_cont[0])
    # print(f"discrete_range: {discrete_range}")
    groups = defaultdict(np.array)
    for d in discrete_range:
        groups[d] = dis_cont[1][dis_cont[0] == d]
    for g in groups:
        H_XY_B += len(groups[g]) / (len(dis_cont[0])) * entropy_estimate_1d_continuous(groups[g], islog2=False)
    print(f"H_XY_B: {H_XY_B}\n")


    H_X = entropy_estimate_1d_discrete(arrs[0])
    print(f"H_X: {H_X}\n")
    p_a = len(dis_dis[0]) /len(arrs[0])
    p_b = len(dis_cont[0]) /len(arrs[0])
    print(f"p_a: {p_a}")
    print(f"p_b: {p_b}")
    MI = p_a * (H_Y_A - H_XY_A) + p_a * (H_Y_B - H_XY_B) + H_X
    print(f"MI: {MI}")
