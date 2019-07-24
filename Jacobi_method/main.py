import matplotlib.pyplot as plt
import numpy as np

def jacobi(A, b, x, n, eps = 1e-15):
    """
        Parameters
        ----------
        A  : list of list of floats : coefficient matrix
        b  : list floats : output vector
        x  : list of floats : initial guess
        eps: float : error tolerance
        n  : max_iteration: int

        Returns
        -------  
        list of floats
            solution to the system of linear equation

        Raises
        ------
        ValueError
            Solution does not converge
    """


    D = np.diag(A)
    R = A - np.diagflat(D) # L + U
    x_last = np.copy(x)

    lost = []

    for step in range(n):
        x = (b - np.dot(R,x))/D
        
        print("step is", step+1,":",x)
        lost.append(np.max(abs(x-x_last)))

        if all(abs(x[i]-x_last[i]) < eps for i in range(len(D))):
            return x, lost
        x_last = np.copy(x)
    return x

if __name__ == '__main__':

    A = np.array([[4.0, 1.0, 1.0, 0.0, 1.0], 
                [-1.0, -3.0, 1.0, 1.0, 0.0], 
                [2.0, 1.0, 5.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, 4.0, 0.0],
                [0.0, 2.0, -1.0, 1.0, 4.0]])
    b = [6.0, 6.0, 6.0, 6.0, 6.0]
    x = [1.0, 1.0, 1.0, 1.0, 1.0]
    n = 100

    x, lost = jacobi(A, b, x, n, 1e-3)
    
    print("solve after", len(lost), "steps:",x)

    # plot
    plt.title("Jacobi's method")
    plt.plot(lost, lw=2)
    plt.ylabel("Lost")
    plt.xlabel("Steps")
    plt.show()