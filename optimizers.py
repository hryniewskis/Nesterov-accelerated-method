import numpy as np
from numpy.linalg import norm


class Nesterov_Optimizers:
    def __init__(self, gamma_u: float = 2, gamma_d: float = 2, lambda_: float = 1.0, like_lasso: bool = True,
                 max_iter: int = 1000):
        assert (gamma_u > 1), "parameter gamma_u has to be greater than 1"
        assert (gamma_d >= 1), "parameter gamma_d has to be greater or equal 1"
        assert (lambda_ >= 0), "parameter lambda_ has to be nonnegative"
        assert (max_iter > 0), "parameter max_iter has to be a positive integer"

        self.lambda_ = lambda_
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.max_iter = max_iter
        self.is_trained = False
        self.like_lasso = like_lasso
        self.A = None
        self.b = None
        self.L = None
        self.coef_ = None

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

    @staticmethod
    def __minimum(a, b, d):
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
        # one can return also
        # return 1 - np.sqrt(2 * A_k * L + 1)
        return 1 + np.sqrt(2 * A_k * L + 1)

    def __gradient_iteration(self, x, M):
        gamma_u = self.gamma_u
        while True:
            T = self.__T_L(x, M)
            if self.__objective(T) <= self.__m_L(x, T, M):
                break
            M = M * gamma_u
        S_L = norm(self.__gradient_f(T) - self.__gradient_f(x)) / norm(T - x)
        return T, M, S_L

    def __gradient_method(self, x, L):
        gamma_d = self.gamma_d
        for it in range(self.max_iter):
            x_prev = x
            x, M = self.__gradient_iteration(x, L)[0:2]
            L = max(L, M * 1.0 / gamma_d)
            # # TODO stop criterion
            if norm(x_prev - x) <= 1e-5:
                print("early stopping at", str(it + 1))
                break
        self.is_trained = True
        self.coef_ = x
        return x

    def __dual_gradient_method(self, x, L):
        lambda_ = self.lambda_

        # To be deleted when implementing appropriate Stoping criterion
        y = x
        # not to be deleted
        psi_a = np.full(shape=x.shape[0], fill_value=0.5)
        psi_b = -x
        psi_d = 0
        for it in range(self.max_iter):
            y_prev = y
            y, M = self.__gradient_iteration(x, L)[0:2]
            L = max(L, M * 1.0 / self.gamma_d)
            psi_b = psi_b + (1. / M) * self.__gradient_f(x)
            psi_d = psi_d + (1. / M) * lambda_
            x = self.__minimum(a=psi_a, b=psi_b, d=psi_d)
            # TODO stop criterion
            if norm(y_prev - y) <= 1e-5:
                print("early stopping at", str(it + 1))
                break
        self.is_trained = True
        self.coef_ = y
        return y

    def __accelerated_method(self, x, L):
        gamma_u = self.gamma_u
        gamma_d = self.gamma_d
        lambda_ = self.lambda_

        psi_a = np.full(shape=x.shape[0], fill_value=0.5)
        psi_b = -x
        # we know that A == psi_d* self.lambda_
        # so we dont need to specify psi_d this time
        A = 0
        v = x
        for it in range(self.max_iter):
            x_prev = x
            while True:
                aL = self.__find_a_times_L(L, A)
                y = (A * x * L + aL * v) / (A * L + aL)
                T_y = self.__T_L(y, L)
                grad_diff = self.__gradient_f(y) - self.__gradient_f(T_y)
                if L * np.dot(grad_diff, y - T_y) >= norm(grad_diff) ** 2:
                    break
                L = L * gamma_u
            A = A + aL / L
            L = L / gamma_d
            x = T_y
            # TODO stop criterion
            if norm(x_prev - x) <= 1e-5:
                print("early stopping at", str(it + 1))
                break
            psi_b = psi_b + aL * self.__gradient_f(x) / L
            v = self.__minimum(a=psi_a, b=psi_b, d=A * lambda_)
        self.is_trained = True
        self.coef_ = x
        return x

    def fit(self, X: np.matrix, y: np.array, method: str = "accelerated"):
        assert y.ndim == 1, "y has to be 1-dimensional"
        assert X.shape[0] == len(y), "number of rows in X has to be the same as length of y"
        assert X.any(), "X cannot be a zero matrix"

        self.A = X
        if self.like_lasso:
            self.lambda_ = self.lambda_ / X.shape[1]
        self.coef_ = np.full(shape=X.shape[1], fill_value=0)
        self.b = y
        self.L = norm(X) ** 2

        if method == "gradient":
            return self.__gradient_method(x=self.coef_, L=self.L)
        elif method == "dual_gradient":
            return self.__dual_gradient_method(x=self.coef_, L=self.L)
        elif method == "accelerated":
            return self.__accelerated_method(x=self.coef_, L=self.L)
        else:
            raise ValueError('wrong argument "method"')

    def predict(self, X):
        if self.is_trained:
            return np.dot(X, self.coef_)
        else:
            raise ValueError("Model isn't trained")

    def get_coef(self):
        if self.is_trained:
            return self.coef_
        else:
            raise ValueError("Model isn't trained")
