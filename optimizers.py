import numpy as np
from numpy.linalg import norm
from numpy.random import random
from sklearn.metrics import mean_squared_error
from math import sqrt


class Nesterov_Optimizers:
    def __init__(self, gamma_u: float = 2, gamma_d: float = 2, lambda_: float = 1.0, like_lasso: bool = True,
                 max_iter: int = 10000, eps=1e-2):
        # TODO add description
        assert (gamma_u > 1), "parameter gamma_u has to be greater than 1"
        assert (gamma_d >= 1), "parameter gamma_d has to be greater or equal 1"
        assert (lambda_ >= 0), "parameter lambda_ has to be nonnegative"
        assert (max_iter > 0), "parameter max_iter has to be a positive integer"

        self.lambda_ = lambda_
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.eps = eps
        self.max_iter = max_iter
        self.is_trained = False
        self.like_lasso = like_lasso
        self.A = None
        self.b = None
        self.L = None
        self.coef_ = None
        self.__name__ = None
        self.history=[]


    def __function_f(self, x):
        return 0.5 * norm(self.b - np.dot(self.A, x)) ** 2

    def __gradient_f(self, x):
        return self.A.T.dot(self.A).dot(x) - self.A.T.dot(self.b)

    def __function_Psi(self, x):
        return self.lambda_ * norm(x, ord=1)

    def __objective(self, x):
        return self.__function_f(x) + self.__function_Psi(x)

    def __m_L(self, y, x, L):
        return (
                self.__function_f(y) + np.dot(self.__gradient_f(y), x - y) +
                (L / 2) * norm(x - y) ** 2 + self.__function_Psi(x)
        )

    # @staticmethod
    def __minimum(self, a, b, d):
        return np.where(
            b < -d,
            -(b + d) / (2 * a),
            np.where(np.logical_and(b > -1 * d, b < d), 0, -(b - d) / (2 * a)),
        )

    def __T_L(self, y, L):
        a = np.full(shape=y.shape[0], fill_value=L / 2)
        b = self.__gradient_f(y) - L * y
        d = np.full(shape=y.shape[0], fill_value=self.lambda_)
        return self.__minimum(a=a, b=b, d=d)

    def __find_a_times_L(self, L, A_k):
        return 1 + np.sqrt(2 * A_k * L + 1)

    def __gradient_iteration(self, x, M):
        gamma_u = self.gamma_u
        while True:
            T = self.__T_L(x, M)
            if self.__objective(T) <= self.__m_L(x, T, M):
                break
            M = M * gamma_u
        # S_L = norm(self.__gradient_f(T) - self.__gradient_f(x)) / norm(T - x)
        return T, M  # , S_L
    
    def __stopping_simple_cond(self,x_current,x_prev,verbose:bool):
        lambda_=self.lambda_
        if verbose:
            #print(norm(x_current-x_prev))
            return norm(x_current-x_prev,ord=2)<self.eps
    
    
    def __stopping_optimal_cond(self, x, verbose: bool):
        lambda_ = self.lambda_
        grad_f = self.__gradient_f(x)
        if np.any(np.abs(x) <= np.finfo(float).eps):
            subgrad_Psi = np.where(np.abs(x) <= np.finfo(float).eps,
                                   -np.sign(grad_f) * np.minimum(np.abs(grad_f), lambda_), np.sign(x) * lambda_)
        else:
            subgrad_Psi = np.sign(x) * lambda_
        if verbose:
            print(norm(grad_f), norm(subgrad_Psi), norm(grad_f + subgrad_Psi))
        return norm(grad_f + subgrad_Psi) <= self.eps

    def __gradient_method(self, x, L, verbose: bool, y_=None):
        gamma_d = self.gamma_d
        for it in range(self.max_iter):
            if it % 1000==0:
                print(it)
            x_prev = x
            x, M = self.__gradient_iteration(x, L)[0:2]
            L = max(L, M * 1.0 / gamma_d)       
            self.history.append(x)
            if self.__stopping_simple_cond(x,x_prev, verbose=verbose):
                print(self.__name__, " early simple stopping at ", it)
                break                  
#             if self.__stopping_optimal_cond(x, verbose=verbose):
#                 print(self.__name__, " early stopping at ", it)
#                 break
        self.is_trained = True
        self.coef_ = x
        return

    def __dual_gradient_method(self, x, L, verbose: bool):
        lambda_ = self.lambda_
        psi_a = np.full(shape=x.shape[0], fill_value=0.5)
        psi_b = -x
        psi_d = 0
        phi_min = np.Inf
        v = x
        for it in range(self.max_iter):
            if it % 1000==0:
                print(it)
            x_prev=x
            y, M = self.__gradient_iteration(v, L)[0:2]
            L = max(L, M * 1.0 / self.gamma_d)
            psi_b = psi_b + (1. / M) * self.__gradient_f(v)
            psi_d = psi_d + (1. / M) * lambda_
            v = self.__minimum(a=psi_a, b=psi_b, d=psi_d)
            # this caused algorithm to stuck at the same time
            # phi_y = self.__objective(y)
            # if phi_min > phi_y:
            #     phi_min = phi_y
            x = y
            self.history.append(x)
            if self.__stopping_simple_cond(x,x_prev, verbose=verbose):
                print(self.__name__, " early simple stopping at ", it)
                break       
#             if self.__stopping_optimal_cond(x, verbose=verbose):
#                 print(self.__name__, " early stopping at ", it)
#                 break             
        self.is_trained = True
        self.coef_ = x
        return

#     def __accelerated_method(self, x, L, verbose: bool):
#         gamma_u = self.gamma_u
#         gamma_d = self.gamma_d
#         lambda_ = self.lambda_

#         psi_a = np.full(shape=x.shape[0], fill_value=0.5)
#         psi_b = -x
#         A = 0
#         v = x
#         for it in range(self.max_iter):
#             while True:
#                 aL = self.__find_a_times_L(L, A)
#                 y = (A * x * L + aL * v) / (A * L + aL)
#                 T_y = self.__T_L(y, L)
#                 grad_diff = self.__gradient_f(y) - self.__gradient_f(T_y)
#                 if np.dot(grad_diff, y - T_y) >= (norm(grad_diff, ord=2) ** 2) / L:
#                     break
#                 L = L * gamma_u
#             A = A + aL / L
#             x = T_y
#             psi_b = psi_b + aL * self.__gradient_f(x) / L
#             v = self.__minimum(a=psi_a, b=psi_b, d=A * lambda_)
#             L = L / gamma_d
#             if self.__stopping_crit_primal(x, verbose=verbose):
#                 print(self.__name__, " early stopping at ", it)
#                 break
#         self.is_trained = True
#         self.coef_ = x
#         return
    def __accelerated_method(self, x, L, verbose: bool):
        gamma_u = self.gamma_u
        gamma_d = self.gamma_d
        lambda_ = self.lambda_        
        psi_a = np.full(shape=x.shape[0], fill_value=0.5)
        psi_b = -x
        psi_d = 0
        A = 0
        v = x
        L = L/gamma_u
        for it in range(self.max_iter):
            if it % 1000==0:
                print(it)
            x_prev=x
            while True:
                L=L*gamma_u
                a=(1+sqrt(2*A*L+1))/L
                y=(A*x+a*v)/(A+a)
                T_y=self.__T_L(y, L)
                grad_diff = self.__gradient_f(y) - self.__gradient_f(T_y)
                if L*np.dot(grad_diff,y-T_y)>norm(grad_diff)**2:
                    break
            x=T_y
            psi_b=psi_b+a*self.__gradient_f(x)
            psi_d=psi_d+a*lambda_
            v=self.__minimum(a=psi_a, b=psi_b, d=psi_d)
            A=A+a
            L=L/(gamma_d*gamma_u)
            self.history.append(x)
            if self.__stopping_simple_cond(x,x_prev, verbose=verbose):
                print(self.__name__, " early simple stopping at ", it)
                break    
#             if self.__stopping_optimal_cond(x, verbose=verbose):
#                 print(self.__name__, " early stopping at ", it)
#                 break
        self.is_trained = True
        self.coef_ = x
        return


    def fit(self, X: np.matrix, y: np.array, method: str = "accelerated", verbose: bool = False):
        # TODO add description
        assert y.ndim == 1, "y has to be 1-dimensional"
        assert X.shape[0] == len(y), "number of rows in X has to be the same as length of y"
        assert X.any(), "X cannot be a zero matrix"

        self.A = X
        self.coef_ = np.full(shape=X.shape[1], fill_value=0)
        if self.like_lasso:
            self.lambda_ = self.lambda_ * X.shape[0]
        self.L = max(norm(X, axis=1)) ** 2
        self.b = y
        self.__name__ = method
        if method == "gradient":
            return self.__gradient_method(x=self.coef_, L=self.L, verbose=verbose)
        elif method == "dual_gradient":
            return self.__dual_gradient_method(x=self.coef_, L=self.L, verbose=verbose)
        elif method == "accelerated":
            return self.__accelerated_method(x=self.coef_, L=self.L, verbose=verbose)
        elif method == "accelerated_plot":
            return self.__accelerated_method_plots(x=self.coef_, L=self.L, y_=y, verbose=verbose)
        if method == "gradient_plot":
            return self.__gradient_method_plots(x=self.coef_, L=self.L, y_=y, verbose=verbose)
        elif method == "dual_gradient_plot":
            return self.__dual_gradient_method_plots(x=self.coef_, L=self.L, y_=y, verbose=verbose)

        else:
            raise ValueError('wrong argument "method"')


    def predict(self, X):
        # TODO add description
        if self.is_trained:
            return np.dot(X, self.coef_)
        else:
            raise ValueError("Model isn't trained")

    def predict_plot(self, X, coef):
        # TODO add description
        print(np.dot(X, coef).shape)
        return np.dot(X, coef)

    def get_coef(self):
        # TODO add description
        if self.is_trained:
            return self.coef_
        else:
            raise ValueError("Model isn't trained")
            
    def get_hist_coef(self):
        if self.history:
            return self.history
        else:
            raise ValueError("Model isn't trained")
            
    def get_objective_value(self):
        if self.history:
            tmp= self.history
        else:
            raise ValueError("Model isn't trained")
        res=[]
        for coef in tmp:
            res.append(self.__objective(coef))
        return res
    
    def get_f_value(self):
        if self.history:
            tmp= self.history
        else:
            raise ValueError("Model isn't trained")
        res=[]
        for coef in tmp:
            res.append(self.__function_f(coef))
        return res


