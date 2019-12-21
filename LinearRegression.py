#####################################
# Linear Regression in one variable #
#####################################

# Gradient Descent         --> gives local minima
# Squared error method     --> gives deviation from actual y values
#############################################################################################
#     theta(i) = theta(i) - alpha * SUMATION ( partialDerivative wrt theta(i) ( J(theta)) ) #
#     partialDerivative wrt theta(0) ( J(theta)) = ( h(X(i)) - Y ) * X(i)                   #
#############################################################################################
# alpha                    --> learning rate, human defined

# Normal Equation          --> directly gives theta values no need for alpha
####################
# ( X'*X )-1 *X'*y #
####################

# Input for the algorithm --> data.csv, no of hours spent studying VS grade


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import math


def partial_derivative_of_theta1(X, y, h) -> float:
    """
    Input:  X, y, h(hypothesis)
    Output: partial derivative of theta1
    """
    dtheta1 = 0
    for i in range(len(y)):                                                                             # iterator over all rows in dataset
        dtheta1 += ((h[i] - y[i]) * X[i])                                                               # formula of partial derivative of J(theta) wrt theta1
    return dtheta1 / len(X)


def partial_derivative_of_theta0(y, h) -> float:
    """
    Input:  X, y, h(hypothesis)
    Output: partial derivative of theta1
    """
    dtheta0 = 0
    for i in range(len(y)):                                                                             # iterator over all rows in dataset
        dtheta0 += (h[i] - y[i])                                                                        # formula of partial derivate of J(theta) wrt theta0
    return dtheta0 / len(y)


def mean_squared_error(y, h) -> float:
    """
    Input:  y, h(hypothesis)
    Output: mean_square_error
    """
    mse = 0
    for i in range(len(y)):                                                                             # iterator over all rows in dataset
        mse += ((h[i] - y[i]) ** 2)                                                                     # formula of mean squared error
    return mse / len(y)


def get_hypothesis(theta0, theta1, X) -> list:
    """
    Input:  X, theta0, theta1
    Output: h
    """
    h = theta0 + theta1 * X                                                                             # formula for calculating hypothesis
    return h


def gradient_descent(theta0, theta1, learning_rate, X, y, h) -> (float, float):
    """
    Input:  theta0, theta1, learning_rate, X, y, h
    Output: new computed values of theta0 and theta1
    """
    dtheta0 = partial_derivative_of_theta0(y, h)
    dtheta1 = partial_derivative_of_theta1(X, y, h)

    theta0 = theta0 - learning_rate * dtheta0
    theta1 = theta1 - learning_rate * dtheta1
    return theta0, theta1


def plot_graph(plt, X, y) -> None:
    """
    Input:  X, y
    Output: points on graph
    """
    plt.scatter(X, y, color="green")


def plot_line(plt, X, h, color, label) -> None:
    """
    Input:  X, h
    Output: line on graph
    """
    plt.plot(X, h, color=color, label=label)


if __name__ == "__main__":

    data = pd.read_csv("data.csv")                                                                      # read input from file
    X = np.array(data['Hours spent Studying'])                                                          # values converts it into a numpy array
    y = np.array(data['Marks scored'])                                                                  # -1 means that calculate the dimension of rows, but have 1 column

    X_train, X_test = X[:80], X[80:]                                                                    # split data to calculate accuracy later
    y_train, y_test = y[:80], y[80:]

    fig = plt.figure()
    fig.tight_layout()
    plt1 = fig.add_subplot(121)
    plt2 = fig.add_subplot(122, projection='3d')
    fig.suptitle("Linear Regression")
    plt1.set_xlabel("Hours Spent Studying")
    plt1.set_ylabel("Marks Scored")
    plt2.set_xlabel("Theta0")
    plt2.set_ylabel("Theta1")
    plt2.set_zlabel("Mean Square Error")
    plot_graph(plt1, X_train, y_train)

    # Gradient Descent Model
    theta0 = 1
    theta1 = 0.1
    learning_rate = 0.0001

    h = get_hypothesis(theta0, theta1, X_train)                                                         # calculate hypothesis for current values of theta
    mse = mean_squared_error(y_train, h)                                                                # calculate mean square error
    plt2.scatter(theta0, theta1, mse, c='r', marker='o')

    #plot_line(X_train, h, "red")
    print(" Theta0: {:.2f}, Theta1: {:.2f}, iteration: {:.2f}, MSE: {:.2f} ".format(theta0, theta1, 0, mse))

    for i in range(1000):
        theta0, theta1 = gradient_descent(theta0, theta1, learning_rate, X_train, y_train, h)           # new values of theta
        h = get_hypothesis(theta0, theta1, X_train)                                                     # calculate hypothesis for current values of theta
        new_mse = mean_squared_error(y_train, h)                                                        # calculate mean square error
        plt2.scatter(theta0, theta1, mse, c='r', marker='o')

        print(" Theta0: {:.2f}, Theta1: {:.2f}, iteration: {:.2f}, MSE: {:.2f} ".format(theta0, theta1, i+1, new_mse))

        if math.isclose(mse, new_mse,abs_tol=1e-05):                                                    # stopping the gradient descent if the mse does not change by more than 1e-05
            print(" Breaking.. Found ideal value of theta0 and theta1.. \n")
            break

        if (new_mse < mse):
            mse = new_mse
        #if i % 20 == 0:
        #    plot_line(plt1, X_train, h, "red")

    plot_line(plt1, X_train, h, color="red", label="Gradient Descent")

    h = get_hypothesis(theta0, theta1, X_test)                                                          # Calculating accuracy on test data
    accuracy = mean_squared_error(y_test, h)
    print("Gradient Descent, theta0: {:.2f}, theta1: {:.2f}, accuracy {:.2f}".format(theta0, theta1, accuracy))

    # Normal Equation Model 
    X_train = np.c_[np.ones(len(X_train)), X_train]                                                     # converting X to (n + 1) dimension
    X_train_transpose = np.transpose(X_train)                                                           # getting X transpose

    theta = np.linalg.inv(X_train_transpose.dot(X_train)).dot(X_train_transpose).dot(y_train)

    h = get_hypothesis(theta[0], theta[1], X_train[:, 1])
    plot_line(plt1, X_train[:, 1], h, color="blue", label="Normal Equation")

    h = get_hypothesis(theta[0], theta[1], X_test)                                                      # Calculating accuracy on test data
    accuracy = mean_squared_error(y_test, h)
    print("Normal Equation, theta0: {:.2f}, theta1: {:.2f}, accuracy {:.2f}".format(theta[0], theta[1], accuracy))

    plt1.legend()
    plt.show()
    