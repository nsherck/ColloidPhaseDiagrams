import math
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
import pickle
import time
import pymbar
# To Do List
# 1) check generality of dN for even dN
# 2) allow more things to be passed as command line arguments
# 3) make graphs save and make more graphs
# 4) double check entropy matrix through an iteration?
# 5) define functions in the text earlier by using lambda functions? or check moving them w/o lambda func.


def sign(x):
    # returns the sign of each element in a list, x
    return [i and (1, -1)[i < 0] for i in x]


def diff(x):
    # returns the changes between subsequent elements of a list, x
    return [x[i + 1] - x[i] for i in range(0, len(x) - 1)]


def chop(x, epsilon=1e-4):
    # returns the list x with very small (< epsilon) elements set to 0
    return [(i, 0)[i < epsilon] for i in x]


def smooth(l, x):
    # smooths data in list l using a moving average with a period of x
    half_x = int(x / 2)
    smth_l = []
    smth_l.extend([sum([l[k] for k in range(0, i)]) / i for i in range(half_x, x)])
    smth_l.extend([sum([l[i + k] / x for i in range(0, x)]) for k in range(0, len(l) - x + 1)])
    smth_l.extend([sum([l[-k] for k in range(i, 0, -1)]) / (i + 1) for i in range(x - 1, half_x, -1)])
    return smth_l


if __name__ == "__main__":
    # For command-line runs, build the relevant parser
    import argparse as ap

    parser = ap.ArgumentParser(description='Perform histogram reweighting from LAMMPS simulations.')
    parser.add_argument('-f', '--file', default='lammps_out.txt', type=str, help='Extracted LAMMPS data as text file.')
    parser.add_argument('-K', '--bins', default=100, type=int, help='Number of energy bins.')
    parser.add_argument('-dN', '--dN', default=3, type=int, help='Number of molecules per bin (typically an odd number.')
    parser.add_argument('-g', '--graphs', default=False, type=bool, help='Make graphs if True.')
    parser.add_argument('-r', '--resolution', default=20, type=int, help='Number of equilibrium points to find.')
    parser.add_argument('-s', '--save', default='s.obj', type=str, help='Filename to store entropy matrix as.')
    parser.add_argument('-l', '--load', default='s.obj', type=str, help='Filename for stored entropy matrix.')

    # Parse the command-line arguments
    args = parser.parse_args()

    file = open(args.file, 'r')  # open data file
    lines = file.readlines()  # read lines of data
    dat = [lines[i].split() for i in range(0, len(lines))]  # split columns into array
    U_ind = dat[0].index('PotEng')  # find column numbers for important parameters
    N_ind = dat[0].index('Atoms')
    T_ind = dat[0].index('v_T_j')
    mu_ind = dat[0].index('v_mu_j')
    V = float(dat[1][dat[0].index('Volume')])  # get volume

    dN = args.dN
    smth_bin = int(0.05 * V / dN)  # set smoothing bin size to include density range of 0.05
    if smth_bin % 2 != 0:  # ensure that this is an even number so smoothed distr. corresponds to N in list of N's
        smth_bin = int(smth_bin + 1)

    dat = dat[1:]  # remove headers from data
    tups = [(float(i[T_ind]), float(i[mu_ind])) for i in dat]  # get tuples of (T,mu) for each data point
    Tmu_pairs = set(tups)  # find distinct pairs
    J = len(Tmu_pairs)  # number of (T,mu) pairs

    key = []  # list for index of T,mu pair
    n = []  # list for number of observations per (T,mu) pair
    for j in Tmu_pairs:
        counter = [i == j for i in tups]  # list of ones where T,mu matches the current (T,mu) pair
        n.append(sum(counter))  # number of matches
        key.append(counter.index(True))  # location of first match
    Tmu_pairs = [i for _, i in sorted(zip(key, Tmu_pairs))]  # reorder Tmu_pairs to be in the order of dat
    n = [i for _, i in sorted(zip(key, n))]  # also reorder the list of observations/state point

    # ordered lists of main relevant parameters
    T = [i for i, _ in Tmu_pairs]
    mu = [i for _, i in Tmu_pairs]
    N = [int(i[N_ind]) for i in dat]
    U = [float(i[U_ind]) * int(i[N_ind]) for i in dat]

    K = args.bins  # rename number of energy bins as K

    if args.graphs:  # graph the distributions of (N,U) points
        for j in range(0, J):
            lbl = 'T = %.3f, $\mu$ = %.3f' % (T[j], mu[j])
            strt = sum([n[i] for i in range(0, j)])
            end = sum([n[i] for i in range(0, j + 1)])
            plt.scatter(N[strt:end], U[strt:end], 0.5, label=lbl)
        plt.legend()
        plt.xlabel('N')
        plt.ylabel('U')
        plt.show(block=False)

    U_max = max(U) + 10 ** (-8)  # include epsilon such that max U value is binned into highest bin
    U_min = min(U)
    N_max = max(N)
    N_min = min(N)
    dU = (U_max - U_min) / K  # energy bin size
    M: int = math.ceil((N_max + 1e-8 - N_min) / dN)  # include epsilon s.t. there's another bin if max N is on cusp
    U_k = [U_min + (k + .5) * dU for k in range(0, K)]  # list of U values corresponding to each bin
    N_k = [N_min + dN * i + (dN - 1) / 2 for i in range(0, M)]  # list of N values corresponding to each bin

    if args.load:  # if specified a file to load entropy matrix
        try:  # try to open the file (in case it does not exist)
            with open(args.load, 'rb') as file2:
                smat = pickle.load(file2)
        except:  # in the event the file does not exist, set s to empty
            s = []
        # if s was loaded properly, check that it is the proper size for the data
        if len(smat) != K or len(smat[0]) != M:
            s = []  # if it does not match the data/bin sizes, set it to be empty
        else:  # if the s matrix matches the data, find where there are nonzero entries
            km_list = []  # get list of locations of nonzero S for building probability distr
            for k in range(0, K):
                for m in range(0, M):
                    if smat[k][m] != 0:
                        km_list.append([k,m])
            s = [smat[k][m] for k, m in km_list]  # reformat matrix into a list
    else:
        s = []

    if not s:  # if s is empty/was not loaded properly, use pymbar to get apprx. free energies, then FS iterations for s
        # reduced potentials for pymbar
        u_kn = [[(U[i] - mu[j] * N[i]) / T[j] for i in range(0, sum(n))] for j in range(0, J)]
        mbar = pymbar.MBAR(u_kn, n)  # perform MBAR method
        results = mbar.getFreeEnergyDifferences()  # find free energy difference matrix
        f1 = [-i for i in results[0][0]]  # get negative differences relative to first state point

        # empty matrix with K rows, M columns for counts of observations
        c = [[0 for i in range(0, M)] for j in range(0, K)]

        f = f1.copy()  # free energies from pymbar
        fnew = [0] * J  # empty vector for iterations/error calculation

        # count (U,N) points in each bin
        for i in range(0, len(dat)):
            k = int((U[i] - U_min) / dU)
            m = int((N[i] - N_min) / dN)
            c[k][m] += 1

        # get list of location for non-zero counts to accelerate Ferrenberg-Swendsen iterations
        km_list = []
        for k in range(0, K):
            m_list = []
            for m in range(0, M):
                if c[k][m] > 0:
                    km_list.append([k, m])

        # iterate using the free energies from pymbar to ensure consistency with bins
        tol = 1e-5  # tolerance for total relative error in free energies
        err = 100 * tol  # initialize error
        count = 0  # counter for number of iterations

        # list of negative reduced potentials
        ukmj = [[(-U_k[k] + mu[j] * N_k[m]) / T[j] for k, m in km_list] for j in range(0, J)]
        log_c = [math.log(c[k][m]) for k, m in km_list]  # list of log of counts
        wj = [0] * J  # list for exponents in log of sum for entropy matrix
        wkm = [0] * len(km_list)  # list for exponents in log of sum for free energies
        s = [0] * len(km_list)  # list of entropies

        while err > tol:  # perform FS iterations
            for i in range(0, len(km_list)):
                for j in range(0, J):
                    wj[j] = ukmj[j][i] - f[j]
                wmax = max(wj)
                s[i] = log_c[i] - wmax - math.log(sum([n[j] * math.exp(wj[j] - wmax) for j in range(0, J)]))
            for j in range(0, J):
                for i in range(0, len(km_list)):
                    wkm[i] = s[i] + ukmj[j][i]
                wmax = max(wkm)
                fnew[j] = (wmax + math.log(sum([math.exp(km - wmax) for km in wkm])))
            print((count, err, fnew))
            fnew[0] = 0
            # calculate error as sum of relative errors of free energies
            err = sum([abs((f[j] - fnew[j]) / (f[j] + 1e-8)) for j in range(0, J)])
            f = [i for i in fnew]
            count += 1
        print(count)
        if args.save:  # save entropy as a matrix
            smat = [[0 for i in range(0, M)] for j in range(0, K)]  # entropy matrix
            for i in range(0, len(km_list)):
                smat[km_list[i][0]][km_list[i][1]] = s[i]
            with open(args.save, 'wb') as file2:
                pickle.dump(smat, file2)

    # define function to give normalized probability distribution given T,mu
    def p(t, Mu):
        p_temp = [[0 for i in range(0, M)] for j in range(0, K)]  # set up empty matrix
        for i in range(0, len(km_list)):  # calculate probabilities
            k = km_list[i][0]
            m = km_list[i][1]
            p_temp[k][m] = math.exp(s[i] - U_k[k] / t + Mu * N_k[m] / t)
        p_tot = sum([sum(k) for k in p_temp])
        return [[j / p_tot for j in i] for i in p_temp]


    def f(t, Mu):  # function which returns error from evenly bimodal for the projected distribution
        p0 = p(t, Mu)  # get 2D prob. distr.
        p_proj = [sum([p0[k][m] for k in range(0, K)]) for m in range(0, M)]  # project onto N
        p_smooth = smooth(p_proj, smth_bin)  # smooth using a moving average
        p_smooth = chop(p_smooth, 1 / M / 10)  # chop small terms (with < 1/10 of uniform prob) for smoothness in savgol
        p_smooth = list(savgol_filter(p_smooth, smth_bin - 1, 1))  # smooth using savgol filter
        der = diff(p_smooth)  # find the difference between adjacent points (proportional to derivative)
        extrma = diff(sign(der))  # find the differences between the signs of the derivative
        peaks = [i for i, x in enumerate(extrma) if x == -2]  # maxima are where derivative switches + to -
        # also look for a peak at the edges since moving average may simply increase if peak is too close to edge
        if der[0] < 0:  # if there is a nonzero, decreasing derivative at the start, this is a peak
            peaks = [0] + peaks
        if der[-1] > 0:  # if there is a nonzero, increasing derivative at the end, this is a peak
            peaks.append(M - 1)
        valleys = [i for i, x in enumerate(extrma) if x == 2]  # minima are where derivative switches - to +
        modes = len(peaks)  # modality is number of peaks
        if modes == 2:  # if there are two peaks
            if len(valleys) == 1:
                valley = valleys[0] + 1
            # should be one minima, but if not take the average of minima between the peaks
            elif len(valleys) > 1:
                valleys = [i for i in valleys if peaks[0] < i < peaks[1]]
                valley = int(sum(valleys) / len(valleys)) + 1
            else:  # if no minima was found, it is likely due to large zero region in between peaks, so use avg of peaks
                valley = int(sum(peaks) / 2) + 1
            sqerr: float = (sum([p_proj[i] for i in range(0, valley)]) - 0.5) ** 2 \
                           + (sum([p_proj[i] for i in range(valley, M)]) - 0.5) ** 2
            return sqerr
        elif modes > 2:  # there cannot be more than two modes, so more should still be counted as bimodal
            valley = int((peaks[0] + peaks[-1]) / 2) + 1  # set the min to be the middle of the two outer peaks
            sqerr: float = (sum([p_proj[i] for i in range(0, valley)]) - 0.5) ** 2 \
                           + (sum([p_proj[i] for i in range(valley, M)]) - 0.5) ** 2
            return sqerr
        else:  # if there was only one peak, return error of 0.5
            return 0.5


    def plot_p(t, Mu):  # plotting function for visual purposes
        p0 = p(t, Mu)
        p_proj = [sum([p0[k][m] for k in range(0, K)]) for m in range(0, M)]
        p_smooth = smooth(p_proj, smth_bin)
        p_smooth = chop(p_smooth, 1 / M / 10)
        p_smooth = list(savgol_filter(p_smooth, smth_bin - 1, 1))
        der = diff(p_smooth)
        extrma = diff(sign(der))
        peaks = [i for i, x in enumerate(extrma) if x == -2]
        if der[0] < 0:
            peaks = [0] + peaks
        if der[-1] > 0:
            peaks.append(M - 1)
        valleys = [i for i, x in enumerate(extrma) if x == 2]
        modes = len(peaks)
        if modes == 2:
            if len(valleys) == 1:
                valley = valleys[0] + 1
            elif len(valleys) > 1:
                valleys = [i for i in valleys if peaks[0] < i < peaks[1]]
                valley = int(sum(valleys) / len(valleys)) + 1
            else:
                valley = int(sum(peaks) / 2) + 1
        elif modes > 2:
            valley = int((peaks[0] + peaks[-1]) / 2) + 1
        plt.scatter(N_k[valley], p_proj[valley], color='k')
        plt.plot(N_k, p_proj)
        plt.plot(N_k, p_smooth)
        plt.show()


    def min_f(t, mumin, mumax):
        # return the equilibrium chemical potential at temperature t, searching between mumin and mumax
        fmin = 0.5  # set the minimum err found to 0.5
        fcut = 0.49  # set the cutoff value for starting the local optimization (sufficiently bimodal cutoff)
        num_pts = int(2 * res)  # search chemical potentials at twice the resolution of plotting
        pts = [mumin + i * (mumax - mumin) / (num_pts - 1) for i in range(0, num_pts)]  # determine mu to check
        new_pts = pts.copy()
        t1 = time.time()
        flag = 0  # flag for whether or not a bimodal distribution has been found
        while fmin > fcut:  # while a bimodal distribution has not been found
            fvals = [0.5] * len(new_pts)  # initialize list of function values
            for i in range(0, len(new_pts)):
                fvals[i] = f(t, new_pts[i])  # for potentials in list of new points to check, calculate error
                if flag == 0:  # if no bimodal distr has been found
                    if fvals[i] < fcut:  # check if the function is bimodal
                        flag = 1  # set flag indicating bimodal distr has been found
                else:  # if bimodal distr has been found
                    if fvals[i] > fcut:  # stop search upon reaching next distr which is not bimodal
                        break
            fmin = min(fvals)  # find minimum err bimodal distr in list
            mu_0 = new_pts[fvals.index(fmin)]  # set initial guess for mu to the one corresponding to said minimum err
            print('Number of bimodal distr. found: ', sum([i < 0.5 for i in fvals]))
            new_pts = [(pts[i] + pts[i + 1]) / 2 for i in range(0, len(pts) - 1)]  # set new pts to btwn old pts
            num_pts = len(pts) + len(new_pts)
            if num_pts > 20 * res and fmin > fcut:  # if the number of points is too high and no bimodal distr. found
                print('No bimodal distribution found.')
                return []
            pts = [mumin + i * (mumax - mumin) / (num_pts - 1) for i in range(0, num_pts)]
        print(time.time() - t1)
        eps = 1e-6  # step size for derivatives
        iters = 0  # counter of iterations
        tol2 = 1e-5  # tolerance for minimization
        mu_s = mu_0  # starting chemical potential
        fval = fmin  # starting err value
        print('The starting error in local minimization is: ', fval)
        # since min is also zero, we use mixture of newton's method for optimization and for root finding
        while fval > tol2 and iters < 100:
            f_plus = f(t, mu_s + eps)  # evaluate necessary info for derivatives
            f_min = f(t, mu_s - eps)
            fder = (f_plus - f_min) / (2 * eps)  # the central finite diff. estimate of 1st derivative
            jac = (f_plus - 2 * fval + f_min) / eps ** 2  # central FD est. of 2nd deriv.
            mu_new1 = mu_s - fval / (fder + eps)  # new mu via root finding
            mu_new2 = mu_s - fder / abs(jac + eps)  # new mu via optimization
            if mu_min < mu_new1 < mu_max and mu_min < mu_new2 < mu_max:  # as long as mu are in given range
                fval1 = f(t, mu_new1)  # evaluate both functions
                fval2 = f(t, mu_new2)
            elif mu_min < mu_new2 < mu_max:  # if only one is in range, evaluate that one only and set other to higher
                fval2 = f(t, mu_new2)
                fval1 = 2 * fval2
            elif mu_min < mu_new1 < mu_max:
                fval1 = f(t, mu_new1)
                fval2 = 2 * fval1
            else:
                raise Exception('Saturation mu expected to be outside given range.')
            if fval1 > fval2:  # set new function value and mu to the one that gives lower value
                mu_s = mu_new2
                fval = fval2
            elif fval1 < fval2:
                mu_s = mu_new1
                fval = fval1
            else:
                if fval1 == 0.5:
                    raise Exception('Bad conditions reached during optimization.')
                else:
                    mu_s = mu_new1
                    fval = fval1
            iters += 1
        print('The optimization took ',iters,' iterations.')
        if iters == 100:
            print('Maximum number of iterations in optimization exceeded.')
            return []
        return mu_s


    def rho_sats(t, Mu):  # function to get the saturation densities and check modality
        p0 = p(t, Mu)
        p_proj = [sum([p0[k][m] for k in range(0, K)]) for m in range(0, M)]
        p_smooth = smooth(p_proj, smth_bin)
        p_smooth = chop(p_smooth, 1 / M / 10)
        p_smooth = list(savgol_filter(p_smooth, smth_bin - 1, 1))
        der = diff(p_smooth)
        extrma = diff(sign(der))
        peaks = [i for i, x in enumerate(extrma) if x == -2]
        if der[0] < 0:
            peaks = [0] + peaks
        if der[-1] > 0:
            peaks.append(M - 1)
        valleys = [i for i, x in enumerate(extrma) if x == 2]
        if len(valleys) == 1:
            valley = valleys[0] + 1
        elif len(valleys) > 1:
            valleys = [i for i in valleys if peaks[0] < i < peaks[1]]
            valley = int(sum(valleys) / len(valleys)) + 1
        else:
            valley = int(sum(peaks) / 2) + 1
        rV = sum([N_k[i] * p_proj[i] for i in range(0, valley)]) / sum([p_proj[i] for i in range(0, valley)]) / V
        rL = sum([N_k[i] * p_proj[i] for i in range(valley, M)]) / sum([p_proj[i] for i in range(valley, M)]) / V
        return [rL, rV, len(peaks)]

    # initialize lists
    rho_L = []  # liquid density
    rho_V = []  # vapor density
    mu_sats = []  # critical chemical potentials
    Ts = []  # critical temperatures
    Mds = []  # modalities
    res = args.resolution  # rename resolution (number of temperature pts to check)
    # create list of temperatures to check
    T_min = min(T)
    T_max = max(T)
    temps = [T_min + i * (T_max - T_min) / (res - 1) for i in range(0, res)]
    # set min and max mu to check within  #### may need to change this ####
    mu_min = min(mu)
    mu_max = max(mu)

    mu0 = mu_min  # rename minimum since range to check will be decreased later
    t0 = time.time()
    for ts in temps:
        mu_sat = min_f(ts, mu0, mu_max)  # find equilibrium chemical potential
        if mu_sat:  # as long as it was found, get critical densities and update lists
            [r_L, r_V, mds] = rho_sats(ts, mu_sat)
            rho_L.append(r_L)
            rho_V.append(r_V)
            mu_sats.append(mu_sat)
            mu0 = mu_sat
            Ts.append(ts)
            Mds.append(mds)
        else:  # if it wasn't found, it should be because T > T of critical pt
            break
    print(time.time() - t0)

    # if modality was not 2 at higher temps, this is likely due to it being past the critical point, so remove these
    for i in range(1, res + 1):
        if Mds[-i] == 2:
            break
        else:
            rho_V.pop(-1)
            rho_L.pop(-1)
            Ts.pop(-1)
            mu_sats.pop(-1)

    if args.graphs:  # plot the data
        plt.plot(rho_V, temps[0:len(rho_V)], color='b', label="Histogram Reweighting")
        plt.plot(rho_L, temps[0:len(rho_L)], color='b')
        # note that the following is specific to LJ fluid
        plt.scatter([1.131E-02, 1.951E-02, 2.560E-02, 3.188E-02, 5.044E-02, 7.951E-02, 1.350E-01],
                    [0.65, 0.70, 0.72871, 0.75, 0.80, 0.85, 0.90], color='k')
        plt.scatter([7.617E-01, 7.293E-01, 7.092E-01, 6.933E-01, 6.521E-01, 6.010E-01, 5.244E-01],
                    [0.65, 0.70, 0.72871, 0.75, 0.80, 0.85, 0.90], color='k', label="NIST")
        plt.legend()
        plt.xlabel('Density, $\\rho$')
        plt.ylabel('Temperature, $T$')
        plt.show(block=False)
