# Re-implementing the function since it was not defined in this session
import numpy as np
from scipy.special import psi  # For digamma function used in the calculation

def discrete_continuous_info(d, c, k=1, base=np.e):
    """
    Estimates the mutual information between a discrete vector 'd' and a continuous vector 'c' using
    nearest-neighbor statistics.

    Parameters:
    d (numpy array): Discrete symbols stored as a 1D array.
    c (numpy array): Continuous data stored as a 2D array with shape (num_dimensions, num_samples).
    k (int): Number of neighbors to consider. Default is 3.
    base (float): Base of the logarithm for calculating mutual information. Default is natural logarithm.

    Returns:
    f (float): Estimated mutual information.
    V (numpy array): Array containing the volume estimates for each sample.
    """
    # Convert inputs to numpy arrays if they are not
    d = np.array(d)
    c = np.array(c)

    # Step 1: Bin the continuous data according to the discrete symbols
    first_symbol = []
    symbol_IDs = np.zeros(d.shape[0], dtype=int)
    c_split = {}
    cs_indices = {}
    num_d_symbols = 0

    # Group columns of c based on the discrete vector d
    for c1 in range(d.shape[0]):
        symbol_IDs[c1] = num_d_symbols + 1
        for c2 in range(num_d_symbols):
            if np.array_equal(d[c1], d[first_symbol[c2]]):
                symbol_IDs[c1] = c2 + 1
                break

        if symbol_IDs[c1] > num_d_symbols:
            num_d_symbols += 1
            first_symbol.append(c1)
            c_split[num_d_symbols] = []
            cs_indices[num_d_symbols] = []

        c_split[symbol_IDs[c1]].append(c[:, c1])
        cs_indices[symbol_IDs[c1]].append(c1)

    # Convert lists to numpy arrays
    for key in c_split:
        c_split[key] = np.array(c_split[key]).T  # Transpose for correct shape

    # Step 2: Compute the neighbor statistic for each data pair (c, d)
    m_tot = 0
    av_psi_Nd = 0
    V = np.zeros(d.shape[0])
    psi_ks = 0

    for c_bin in range(1, num_d_symbols + 1):
        one_k = min(k, c_split[c_bin].shape[1] - 1)
        if one_k > 0:
            for pivot in range(c_split[c_bin].shape[1]):
                # Calculate distances within the same discrete symbol group
                vector_diff = c_split[c_bin] - c_split[c_bin][:, pivot:pivot+1]
                c_distances = np.linalg.norm(vector_diff, axis=0)

                sorted_distances = np.sort(c_distances)
                eps_over_2 = sorted_distances[one_k]  # kth nearest neighbor

                # Calculate distances using all samples in c
                vector_diff_all = c - c_split[c_bin][:, pivot:pivot+1]
                all_c_distances = np.linalg.norm(vector_diff_all, axis=0)

                m = max(np.sum(all_c_distances <= eps_over_2) - 1, 0)  # Don't count the pivot itself

                m_tot += psi(m)
                V[cs_indices[c_bin][pivot]] = (2 * eps_over_2) ** c.shape[0]
        else:
            m_tot += psi(num_d_symbols * 2)

        p_d = c_split[c_bin].shape[1] / d.shape[0]
        av_psi_Nd += p_d * psi(p_d * d.shape[0])
        psi_ks += p_d * psi(max(one_k, 1))

    f = (psi(d.shape[0]) - av_psi_Nd + psi_ks - m_tot / d.shape[0]) / np.log(base)
    # print("psi(d.shape[0]): ", psi(d.shape[0]))
    # print("m_tot / d.shape[0]: ", m_tot / d.shape[0])
    # print("av_psi_Nd: ", av_psi_Nd)
    # print("psi_ks: ", psi_ks)
    # print("mean of log V: ", np.mean(np.log(V)))
    # print("entropy: ", psi(d.shape[0])- m_tot / d.shape[0]+np.mean(np.log(V)))
    return f, V

#
# Generate 1000 samples where the discrete values can only take the value 1
# d = np.ones(1000)
#
# # Generate 1000 samples of continuous data randomly and uniformly sampled from [0, 1]
# c = np.random.uniform(0, 1, size=(1, 1000))  # 1D continuous data for simplicity
#
# # Call the discrete_continuous_info function
# mi, volume = discrete_continuous_info(d, c, k=3)
#
# print(mi)

