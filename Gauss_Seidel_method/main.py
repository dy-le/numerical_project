import matplotlib.pyplot as plt
import numpy as np

def gauss_seidel(m, b, x0=None, eps=1e-3, max_iteration=100):
    """
    Parameters
    ----------
    m  : list of list of floats : coefficient matrix
    x0 : list of floats : initial guess
    eps: float : error tolerance
    max_iteration: int
    
    Returns
    -------  
    list of floats
        solution to the system of linear equation
    
    Raises
    ------
    ValueError
        Solution does not converge
    """
    n  = len(m)
    
    D = np.diag(m)
    R = m - np.diagflat(D)

    x0 = np.zeros(n) if x0 == None else x0
    x_last = np.copy(x0)
    x = np.copy(x0)

    lost = []

    for step in range(max_iteration):
        for i in range(n):
            x[i] = sum(-R[i]*x)/D[i] + b[i]/D[i]
        
        print("step is", step+1,":",x)
        lost.append(np.max(abs(x-x_last)))

        if all(abs(x[i]-x_last[i]) < eps for i in range(n)):
            return x, lost
        x_last = np.copy(x)  
    raise ValueError('Solution does not converge')

if __name__ == '__main__':
    m = np.array([[4.0, 1.0, 1.0, 0.0, 1.0], 
                [-1.0, -3.0, 1.0, 1.0, 0.0], 
                [2.0, 1.0, 5.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 4.0, 0.0],
                [0.0, 2.0, -1.0, 1.0, 4.0]])
    b = np.array([6.0, 6.0, 6.0, 6.0, 6.0])
    x = [1.0, 1.0, 1.0, 1.0, 1.0]
    n = 100

    x, lost = gauss_seidel(m,b,max_iteration=n)

    print("solve after", len(lost), "steps:",x)

    # plot
    plt.title("Gauss-Seidel's method")
    plt.plot(lost, lw=2)
    plt.ylabel("Lost")
    plt.xlabel("Steps")
    plt.show()
