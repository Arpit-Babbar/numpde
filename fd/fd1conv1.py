"""
Convergence of finite difference approximation to first derivative using
backward, forward and central difference.
              u(x) = sin(2*pi*x), x in [0,1]
"""
import numpy as np
import matplotlib.pyplot as plt

xmin, xmax = 0.0, 1.0

def fun(x):
    return np.sin(2.0*np.pi*x)
    
def dfun(x):
    return 2.0*np.pi*np.cos(2.0*np.pi*x)

def ddfun(x):
    return -(2.0*np.pi)**2 * np.sin(2.0*np.pi*x)

npt = [50,100,200,400,800]
errb = []; errf = []; errc = []; hh=[]
for n in npt:
    x = np.linspace(xmin,xmax,n)
    h = (xmax - xmin)/(n-1); hh.append(h)
    f = fun(x)
    # backward difference
    dfb = (f[1:-1] - f[0:-2])/h
    # forward difference
    dff = (f[2:] - f[1:-1])/h
    # central difference
    dfc = (f[2:] - f[0:-2])/(2*h)
    # exact derivative
    dfe= dfun(x[1:-1])
    # errors
    errb.append(np.abs(dfe - dfb).max())
    errf.append(np.abs(dfe - dff).max())
    errc.append(np.abs(dfe - dfc).max())
    
# Compute convergence rates
print("%5s %17s %10s %10s" % ('h','Backward','Forward','Central'))
for i in range(1,len(hh)):
    pb = np.log(errb[i-1]/errb[i])/np.log(2)
    pf = np.log(errf[i-1]/errf[i])/np.log(2)
    pc = np.log(errc[i-1]/errc[i])/np.log(2)
    print("%e   %f   %f   %f" % (hh[i],pb,pf,pc))

# Plot error convergence
plt.loglog(hh,errb,'o-',hh,errf,'*-',hh,errc,'s--')
plt.xlabel('h')
plt.ylabel('Maximum error')
plt.legend(('Backward','Forward','Central'),loc='lower right')
plt.title('First derivative for $u(x)=\sin(2\pi x)$, $x\in[0,1]$')
plt.show()
