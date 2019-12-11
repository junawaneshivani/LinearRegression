#######################################
## Linear Regression in one variable ##
#######################################

## Squared error method     --> gives deviation from acutal y values
## Gradient Descent         --> gives local mimima
###############################################################################################
###     theta(i) = theta(i) - alpha * SUMATION ( partialDerivate wrt theta(i) ( J(theta)) ) ###
###     partialDerivate wrt theta(0) ( J(theta)) = ( h(X(i)) - Y ) * X(i)                   ###
###############################################################################################
## alpha                    --> learning rate, human defined
## Input for the algorithm --> data.csv, no of hours spent studying VS grade 


import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np
import math              as math
import matplotlib.style  as style
#style.use("ggplot")


def hypothesis(theta0, theta1, x_i ) -> float :
    """
    Compute value of hypothese h(theta) for X(i) 
    Input:  theta0, theta1, X(i)
    Output: h(X(i))
    """
    return theta0 + x_i * theta1                                         # calculate hypothesis as per formula


def partial_derivative_of_theta1(X, y, y_pred ) -> float:
    """
    Input:  X, y, y_pred
    Ouptut: mean_square_error
    """
    sum = 0
    for i in range(len(y)):                                             # iterator over all rows in dataset
        sum += ( (y_pred[i] - y[i]) * X[i]  )                           # formula of partial derivate of J(theta) wrt theta1
    return sum / len(X)                                                         

def partial_derivative_of_theta0(y, y_pred ) -> float:
    """
    Input:  X, y, y_pred
    Ouptut: mean_square_error
    """
    sum = 0
    for i in range(len(y)):                                             # iterator over all rows in dataset
        sum +=  (y_pred[i] - y[i])                                      # formula of partial derivate of J(theta) wrt theta0
    return sum  / len(y)                                                         

def mean_squared_error(y, y_pred ) -> float:
    """
    Input:  y, y_pred
    Ouptut: mean_square_error
    """
    sum = 0
    for i in range(len(y)):                                             # iterator over all rows in dataset
        sum += ( (y_pred[i] - y[i]) ** 2 )                              # formula of partial derivate of J(theta) wrt theta0  
    return sum / len(y)                                                        

def get_predicted_y(theta0, theta1, X) -> list:
    """
    Input:  X, theta0, theta1
    Output: y_pred
    """
    y_pred = np.empty_like(X)
    for i in range(len(X)):
        y_pred[i] = hypothesis(theta0, theta1, X[i] )
    
    return y_pred

def gradient_descent(theta0, theta1, learning_rate, X, y, y_pred) -> (float,float):
    """
    Input:  theta0, theta1, learning_rate, X, y, y_pred
    Output: new computed values of theta0 and theta1
    """
    dtheta0 = partial_derivative_of_theta0(y, y_pred)
    dtheta1 = partial_derivative_of_theta1(X, y, y_pred)
    
    theta0  = theta0 - learning_rate * dtheta0
    theta1  = theta1 - learning_rate * dtheta1
    return theta0, theta1

def plot_graph_and_line(X, y, y_pred) -> None:
    """
    Input:  X, y, y_pred
    Output: points and lines on graph
    """
    plt.title("Linear Regression")
    plt.scatter(X, y, color="green")
    plt.plot(X, y_pred, color="red")
    plt.show(block=False)

if __name__ == "__main__":
    
    data = pd.read_csv("data.csv")                                      # read input from file
    X    = np.array( data['Hours spent Studying'])                      # values converts it into a numpy array
    y    = np.array( data['Marks scored'])                              # -1 means that calculate the dimension of rows, but have 1 column

    X_train, X_test = X[:80], X[80:]                                    # split data to calculate accuracy later
    y_train, y_test = y[:80], y[80:]

    theta0        = 10
    theta1        = 0.1
    learning_rate = 0.0001
    
    y_pred = get_predicted_y(theta0, theta1, X_train)                   # calculate y_pred for current values of theta
    mse = mean_squared_error(y_train, y_pred)                           # calculate mean square error
    plot_graph_and_line(X_train, y_train, y_pred)
    
    for i in range(1000):
        theta0, theta1 = gradient_descent(theta0, theta1, learning_rate, X_train, y_train, y_pred)  # new values of theta
        y_pred = get_predicted_y(theta0, theta1, X_train)                                           # calculate y_pred for current values of theta
        new_mse = mean_squared_error(y_train, y_pred)                                               # calculate mean square error
        
        print (" Theta0: {}, Theta1: {}, iteration: {}, MSE: {} ".format(theta0, theta1, i, new_mse))
        
        if math.isclose(mse, new_mse, abs_tol=1e-05):                   # stopping the gradient descent if the mse does not change by more than 1e-05
            print(" Breaking.. Found ideal value of theta0 and theta1.. ")
            plot_graph_and_line(X_train, y_train, y_pred)
            break       
        
        if (new_mse < mse):
            mse = new_mse    
        if i % 20 == 0:
            plot_graph_and_line(X_train, y_train, y_pred)
    
    y_pred = get_predicted_y(theta0, theta1, X_test)                    # Calculating accuracy on test data
    accuracy = "Accuracy: " + str(mean_squared_error(y_test, y_pred))
    
    plt.text(30, 120, accuracy, bbox=dict(facecolor='red', alpha=0.5))    
    plt.show()
    