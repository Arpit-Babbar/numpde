"""
Solve u_t + f(u)_x = 0  for f(u) = u^2/2
Finite volume scheme
Smooth initial data
"""
import numpy as np

#We have put Dirichlet bc on first boundary point, so we don't update
#it and thus do not compute the flux there. This already fixes the last
#boundary point, so we do not need to specify any bc there.

# Burgers flux
def flux_b(u):
    return 0.5*u*u

# Lax-Friedrich flux
def flux_lf(lam,u):
    n  = len(u)
    f  = flux_b(u)
    nf = np.zeros(n+1)
    for i in range(1,n+1):
        nf[i] = 0.5*(f[i-1] + f[i]) - 0.5*(u[i] - u[i-1])/lam
    return nf

# Local Lax-Friedrich flux
def flux_llf(lam,u):
    #lambda unused here
    n  = len(u)
    f  = flux_b(u)
    nf = np.zeros(n+1)
    for i in range(1,n+1):
        a     = np.abs([u[i-1], u[i]]).max()
        nf[i] = 0.5*(f[i-1] + f[i]) - 0.5*a*(u[i] - u[i-1])
    return nf

# Lax-Wendroff flux
def flux_lw(lam,u):
    n  = len(u)
    f  = flux_b(u)
    nf = np.zeros(n+1)
    for i in range(1,n+1):
        a     = 0.5*(u[i-1]+u[i])
        nf[i] = 0.5*(f[i-1] + f[i]) - 0.5*lam*a*(f[i] - f[i-1])
    return nf

# Roe flux
def flux_roe(lam,u):
    #lambda unused here
    n  = len(u)
    f  = flux_b(u)
    nf = np.zeros(n+1)
    for i in range(1,n+1):
        a     = np.abs(0.5*(u[i-1]+u[i]))
        nf[i] = 0.5*(f[i-1] + f[i]) - 0.5*a*(u[i] - u[i-1])
    return nf

# Roe flux with entropy fic
def flux_eroe(lam,u):
    #lambda unused here
    n  = len(u)
    f  = flux_b(u)
    nf = np.zeros(n+1)
    for i in range(1,n+1):
        delta = 0.5*np.abs(u[i]-u[i-1]) if u[i-1] < u[i] else 0.0
        a     = np.abs(0.5*(u[i-1]+u[i]))
        a     = delta if a < delta else a
        nf[i] = 0.5*(f[i-1] + f[i]) - 0.5*a*(u[i] - u[i-1])
    return nf

# Godunov flux
def flux_god(lam,u):
    #lambda unused here
    n  = len(u)
    nf = np.zeros(n+1)
    for i in range(1,n):
        u1 = max(0.0, u[i-1])
        u2 = min(0.0, u[i])
        nf[i] = max(flux_b(u1), flux_b(u2))
    #On the last face, we use only one value of solution.
    nf[n] = flux_b(u[n-1])
    return nf