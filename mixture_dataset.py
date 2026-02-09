import numpy as np

def get_mixture(num_samples=100):
    # produce samples of (x,y)
    # x is discrete-valued and y is mixture
    discrete_y = [-1, 0, 1] # y can be sampled from discrete_y or a normal distribution
    p_discrete = 0.5  # prob that y is sampled from discrete_y
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    for i in range(num_samples):
        rdn = np.random.uniform(0, 1, 1)
        if rdn < p_discrete:
            which = len(discrete_y) * np.random.uniform(0, 1, 1)
            which = int(which)
            y[i] = discrete_y[which]
            x[i] = 1
        else:
            y[i] = np.random.normal(0, 1, 1)
            x[i] = 0
    return x,y

def discrete_flag(arr):
    unique_set = set()
    multiple_set = set()
    for e in arr:
        if e in unique_set:
            unique_set.remove(e)
            multiple_set.add(e)
        elif e in multiple_set:
            continue
        else:
            unique_set.add(e)
    discrete = np.ones(len(arr))
    for i, e in enumerate(arr):
        if e in unique_set:   
            discrete[i] = 0
    return discrete
    

def unmix(arrs):
    flag  = discrete_flag(arrs[1])
    dis_dis = (arrs[0][flag == 1], arrs[1][flag == 1])   
    dis_cont = (arrs[0][flag == 0], arrs[1][flag == 0])   
    return dis_dis, dis_cont
    

if __name__ == "__main__":
    num_samples = 100
    arrs = get_mixture(num_samples)
    # for i in range(num_samples):
    #     print(arrs[0][i], arrs[1][i])
    flag = discrete_flag(arrs[1])
    # for i in range(num_samples):
    #     print(arrs[0][i]-flag[i])
    dis_dis, dis_cont = unmix(arrs)
    for i in range(len(dis_dis[0])):
        print(dis_dis[0][i], dis_dis[1][i])
    for i in range(len(dis_cont[0])):
        print(dis_cont[0][i], dis_cont[1][i])
   









