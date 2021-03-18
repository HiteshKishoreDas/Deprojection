# %%

import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate as intg


def real_fr(r):
    """
    The real radial density profile
    """

    r0 = 1.0
    rcut = 2.0
    if r < rcut:
        return np.exp(-r/r0)
    else:
        return 0.0


def proj(fr, d, x, y):
    """
    To calculate the projected density value at any given x and y
    """

    R = np.sqrt(x*x+y*y)

    def integ(theta):
        return (R/np.cos(theta)**2) * fr(R/np.cos(theta))

    if R != 0:
        return intg.quad(integ, -np.pi/2, np.arctan(d/R))[0]
    else:
        return 2.0*intg.quad(fr, 0, d)[0]


def proj_image(N, d_box, D):
    """
    To create the projected density image
    """

    proj_array = np.zeros((N, N), dtype=float)
    i_mid = int(N/2)
    j_mid = int(N/2)
    for i in range(N):
        for j in range(N):
            x = (j-j_mid)*d_box/N
            y = (i_mid-i)*d_box/N

            proj_array[i, j] = proj(real_fr, D, x, y)

    return proj_array


def proj_r(proj_arr, d_N, d_box):
    """
    To caculate the theta-averaged projection radial profile
    """

    dr = d_box/d_N
    N = np.shape(proj_arr)[0]
    n_r = int(0.5*d_box/dr)
    i_mid = int(N/2)
    j_mid = int(N/2)

    rad_prof = np.zeros(n_r, dtype=float)
    r_arr = np.zeros_like(rad_prof)

    for n in range(n_r-1, -1, -1):
        r_a = dr*n
        r_b = dr*(n+1)

        s = 0.0
        count = 0
        for i in range(N):
            for j in range(N):
                x = (j-j_mid)*d_box/N
                y = (i_mid-i)*d_box/N
                r2 = x*x + y*y

                if r_a*r_a <= r2 < r_b*r_b:
                    s += proj_arr[i, j]
                    count += 1

        if count != 0:
            rad_prof[n] = s/count

        r_arr[n] = (r_b+r_a)*0.5

    return rad_prof, r_arr


def change(rad_prof, r_arr):
    """
    To calculate rate of change in a quantity
    """

    N = np.size(rad_prof)
    drad = np.zeros(N-2, dtype=float)

    for i in range(1, N-2, 1):
        drad[i-1] = (rad_prof[i+1]-rad_prof[i-1])/(r_arr[i+1]-r_arr[i-1])

    return drad


def radius(rad_prof, r_arr):
    """
    To calculate the radius of the feature
    """

    dr2 = change(change(rad_prof, r_arr), r_arr[1:-1])

    return r_arr[2 + np.argmax(dr2)]


def eqn_coeff (r_arr,r):

    r_N = np.size(r_arr)

    coeff_list = []

    r_bnd = np.zeros(r_N+1,dtype=float)
    r_bnd[1:r_N+1] = r_arr+0.5*(r_arr[1]-r_arr[0])


    for i in range(r_N,0,-1):

        if r_bnd[i-1]<r:
            if r_bnd[i]>r:
                coeff = 2*( np.sqrt(r_bnd[i]*r_bnd[i] - r*r ) )
            else:
                coeff = 0.0

        else:
            coeff = 2*( np.sqrt(r_bnd[i]*r_bnd[i] - r*r ) - np.sqrt(r_bnd[i-1]*r_bnd[i-1] - r*r ) )

        coeff_list.append(coeff)

    return np.array(coeff_list)


def deproj(rad_prof, r_arr):
    """
    To calculate the deprojected radial density profile
    """

    coeff_arr = []

    for r in r_arr:

        coeff_arr.append(eqn_coeff(r_arr,r))

    coeff_arr = np.array(coeff_arr)

    rho_dep = np.linalg.solve(coeff_arr,rad_prof)

    return rho_dep[::-1]




N = 40  # Linear pixel density
d_N = 40 # Double of array size of radial profile
d_box = 6.0  # Box length
D = 10.0  # distance of observer from the feature
0
proj_array = proj_image(N, d_box, D)

dep, r = proj_r(proj_array, d_N, d_box)

# print(radius(dep, r))

# coeff_arr = eqn_coeff(r, r[29])

rho_dep = deproj(dep,r)

# print(coeff_arr)

# %%

plt.imshow(proj_array)
plt.colorbar()
# print(proj(real_fr, 10, 0.0, 0.0))

plt.figure()
plt.plot(r, dep)

plt.figure()
plt.plot(r, rho_dep)
plt.plot(r, np.vectorize(real_fr)(r))

# %%
