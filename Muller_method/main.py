import math
import matplotlib.pyplot as plt
import numpy as np

def swap_points(x):
    s = x
    s.sort()
    f = s[1]
    sn = s[2]
    t = s[0]
    s[0] = f
    s[1] = sn
    s[2] = t
    return s

def mullers_method(func, p0, p1, p2, max_iter = 100, eps = 1e-3):
    '''Approximate solution of f(x)=0 by Muller's method.

    Parameters
    ----------
    func : function
        Function for which we are searching for a solution f(x)=0.
    p0, p1, p2 : number
        Parameters init
    eps : number
        Error tolerance, stopping criteria is abs(f(x)) < eps.
    max_iter : integer
        Maximum number of iterations of Muller's method.

    Returns
    -------
    xn : number
        Implement Muller's method: To find a solution to f (x) = 0 
        given three approximations p0 , p1 , and p2.
        Continue until abs(f(root)) < eps and return xn.
        If the number of iterations exceeds max_iter, then return None.
    
    ls : array[step]

    Examples
    --------
    >>> func = lambda x: x**5 - x**4 + 2*x**3 - 3*x**2 + x - 4
    >>> mullers_method(func, 0.0, 1.0, 2.0)
    Found solution after 9 iterations.
    1.4981900316451104
    '''
    
    x = [p0,p1,p2]
    ls = []

    for loopCount in range(max_iter):
        x = swap_points(x)
        y = func(x[0]), func(x[1]), func(x[2])
        h1 = x[1]-x[0]
        h2 = x[0]-x[2]
        lam = h2/h1
        c = y[0]
        a = (lam*y[1] - y[0]*((1.0+lam))+y[2])/(h1 + h2)
        b = (y[1] - y[0] - a*((h1)**2.0))/(h1)
        
        if b > 0:
            root = x[0] - ((2.0*c)/(b+ (b**2 - 4.0*a*c)**0.5))
        else:
            root = x[0] - ((2.0*c)/(b- (b**2 - 4.0*a*c)**0.5))

        ls.append(root)
        if(abs(func(root))<eps):
            return root, ls
        print ("a = %.9f b = %.9f c = %.9f root = %.9f " % (a,b,c,root))
        print ("Current approximation at %d is %.15f" % (loopCount+1,root))
        
        if abs(func(root)) > x[0]:
            x = [x[1],x[0],root]
        else:
            x = [x[2],x[0],root]
        x = swap_points(x)


if __name__ == '__main__':
    func = lambda x: x**5 - x**4 + 2*x**3 - 3*x**2 + x - 4
    root, ls = mullers_method(func, 0.0, 1.0, 2.0)
    point = [func(x) for x in ls]
    init = [0,1,2]
    point0 = [func(x) for x in init]
    # plot
    plt.title("Muller's method")
    plt.text(0.0, 100, 'f(x) = x**5 - x**4 + 2*x**3 - 3*x**2 + x - 4')
    t1 = np.arange(-3.0, 3.0, 0.01)
    plt.plot(t1, func(t1), lw=2)
    plt.plot(ls, point, 'ro')
    plt.plot(init, point0, 'bo')
    plt.annotate('approximation', xy=(ls[-1], point[-1]), xytext=(2, -50),
             arrowprops=dict(facecolor='black', shrink=0.005),
             )
    plt.xlim(-0.5,2.5)
    plt.ylim(-150,150)
    plt.show()
