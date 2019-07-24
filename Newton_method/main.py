import numpy as np
import matplotlib.pyplot as plt

def newton(f,Df,x0,eps,max_iter):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    eps : number
        Error tolerance, stopping criteria is abs(f(x)) < eps.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < eps and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    
    ls : array[step]

    Examples
    --------
    >>> f = lambda x: x**5 - x**4 + 2*x**3 - 3*x**2 + x - 4
    >>> Df = lambda x: 5*x**4 - 4*x**3 + 6*x**2 - 6*x + 1
    >>> newton(f,Df,0,1e-4,100)
    Found solution after 9 iterations.
    1.4981900316451104
    '''
    xn = x0
    ls = []
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < eps:
            print('Found solution after',n,'iterations.')
            return xn, ls
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
        print("Step",n+1,"is :",xn)
        ls.append(xn)
    print('Exceeded maximum iterations. No solution found.')
    return None

if __name__ == "__main__":
    f = lambda x: x**5 - x**4 + 2*x**3 - 3*x**2 + x - 4
    Df = lambda x: 5*x**4 - 4*x**3 + 6*x**2 - 6*x + 1
    approx, ls = newton(f,Df,0,1e-4,100)
    print(approx)


    t1 = np.arange(-10.0, 10.0, 0.1)
    # plot
    plt.figure(figsize=(9,9))
    plt.title("Newton's method")
    
    plt.subplot(221)
    plt.ylabel("f(x)")
    plt.plot(t1, f(t1), lw=2)
    point = [f(x) for x in ls]
    plt.plot(ls ,point, 'bo')
    plt.plot(ls[-1],point[-1], 'go')

    plt.subplot(222)
    plt.ylabel("Df(x)")
    plt.plot(t1, Df(t1), lw=2)
    point = [Df(x) for x in ls]
    plt.plot(ls ,point, 'bo')
    plt.plot(ls[-1],point[-1], 'go')
    
    plt.subplot(223)
    plt.ylabel("Df(x)")
    point = [Df(x) for x in ls]
    plt.plot(point, 'bo')
    plt.plot(8,point[-1], 'go')

    plt.subplot(224)
    plt.ylabel("x")
    plt.plot(ls, 'ro')
    plt.plot(8,ls[-1], 'go')

    plt.show()
