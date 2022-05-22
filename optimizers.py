import numpy as np
from numpy.linalg import norm


class Nesterov_Optimizers:
    def __init__(self, gamma_u: float = 2, gamma_d: float = 2, lambda_: float = 1.0, max_iter: int = 1000):
        assert (gamma_u > 1), "parameter gamma_u has to be greater than 1"
        assert (gamma_d >= 1), "parameter gamma_d has to be greater or equal 1"
        assert (lambda_ >= 0), "parameter lambda_ has to be nonnegative"
        assert (max_iter > 0), "parameter max_iter has to be a positive integer"

        self.lambda_ = lambda_
        self.gamma_u = gamma_u
        self.gamma_d = gamma_d
        self.max_iter = max_iter
        self.is_trained = False
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

    def __objective_subgradient(self, x, L):
        T = self.__T_L(x, L)
        return L * (x - T) + self.__gradient_f(T) - self.__gradient_f(x)

    def __m_L(self, y, x, L):
        # assert y.shape[0] == x.shape[0], "y and x have to be in the same shape"
        return (
                self.__function_f(y) + np.dot(self.__gradient_f(y), x - y) +
                (L / 2) * norm(x - y) ** 2 + self.__function_Psi(x)
        )

    @staticmethod
    def __minimum(a, b, d):
        # TODO sprawdzanie a>0 i d>0
        #         assert np.all(a > 0), "all elements of vector a have to be positive"
        #         assert np.all(d >= 0), "all elements of vector d have to be nonnegative"

        return np.where(
            b < -d,
            -(b + d) / (2 * a),
            np.where(np.logical_and(b > -1 * d, b < d), 0, -(b - d) / (2 * a)),
        )

    def __T_L(self, y, L):
        # assert L > 0, "L has to be greater than 0"

        # a = np.array([L / 2 for i in range(y.shape[0])])
        a = np.full(shape=y.shape[0], fill_value=L / 2)
        b = self.__gradient_f(y) - L * y
        # d = np.array([self.lambda_ for i in range(y.shape[0])])
        d = np.full(shape=y.shape[0], fill_value=self.lambda_)
        return self.__minimum(a=a, b=b, d=d)

    def __gradient_iteration(self, x, M):
        L = M
        T = self.__T_L(x, L)
        while self.__objective(T) > self.__m_L(x, T, L):
            L = L * self.gamma_u
            T = self.__T_L(x, L)
        S_L = norm(self.__gradient_f(T) - self.__gradient_f(x)) / norm(T - x)
        return T, L, S_L

    def __gradient_method(self, y, L):
        L_k = L
        stop_criterion = False
        for it in range(self.max_iter):
            y_prev = y
            y, M = self.__gradient_iteration(y, L_k)[0:2]
            L_k = max(L, M * 1.0 / self.gamma_d)
            # # TODO warunek stopu
            if norm(y_prev - y) <= 1e-5:
                print("early stopping at", str(it + 1))
                break
            print("iteration ",it,"objective function value ",self.__objective(y))
        self.is_trained = True
        return y

    def __dual_gradient_method(self, v, L):
        # do wywalenia przy poprawnym warunku stopu
        y = v
        # dalej to juz zostaje
        L_k = L
        psi_a = np.full(shape=v.shape[0], fill_value=0.5)
        psi_b = -v
        psi_d = 0
        for it in range(self.max_iter):
            y_prev = y
            y, M = self.__gradient_iteration(v, L_k)[0:2]
            L_k = max(L, M * 1.0 / self.gamma_d)
            psi_b = psi_b + (1. / M) * self.__gradient_f(v)
            psi_d = psi_d + (1. / M) * self.lambda_
            v = self.__minimum(a=psi_a, b=psi_b, d=psi_d)
            # # TODO warunek stopu
            if norm(y_prev - y) <= 1e-5:
                print("early stopping at", str(it + 1))
                break
            print("iteration ",it,"objective function value ",self.__objective(y))
        self.is_trained = True
        return y

    def __quadratic_equation(self, L, A_k):
        assert L != 0, "L has to be different than 0"
        return (1 + np.sqrt(2 * A_k * L + 1)) / L

    def __accelerated_method(self, x, L, coef):
        psi_a = np.full(shape=x.shape[0], fill_value=0.5)
        psi_b = -x
        # we know that A_k == psi_d
        A_k = 0
        L_k = L
        x_k = x
        v_k = x
        for it in range(self.max_iter):
            #print(f"Iteration {it}")
            L = L_k
            #print(x_k)
            while True:
                a = self.__quadratic_equation(L, A_k)
                #print(f"y before update: {v_k}")
                #print(f"(A_k * x_k) before update: {(A_k * x_k)}")
                #print(f"a before update: {a}")
                #print(f"A_k before update: {A_k}")
                #print(f"A_k + a before update: {A_k + a}")


                y = ((A_k * x_k) + a * v_k) / (A_k + a)
                #print(f"y is after update to: {y}")
                T_L_y = self.__T_L(y, L)
                obj_sub_T_L = self.__objective_subgradient(T_L_y,L)
                #print("first part:", (np.dot(obj_sub_T_L, y - T_L_y)), "Second Part ", (1 / L) * norm(obj_sub_T_L) ** 2)
                if (np.dot(obj_sub_T_L, y - T_L_y) < (1 / L) * norm(obj_sub_T_L) ** 2):
                    #print(f"L is equal to:{L}")
                    L = L * self.gamma_u
                else:
                    break
            y_k = y
            M_k = L
            A_k = A_k + a
            L_k = M_k / self.gamma_d
            x_k = T_L_y
            #print(f"psi_b before update: {psi_b}")
            psi_b = psi_b + a * self.__gradient_f(x_k)
            #print(f"psi_b after update: {psi_b}")
            #print(f"self.__gradient_f(x_k) {self.__gradient_f(x_k)}")

            v_k = self.__minimum(a=psi_a, b=psi_b, d=A_k*self.lambda_)
            print(norm(x_k - coef))
        return x_k

    def fit(self, X: np.matrix, y: np.array, coef, method: str = "gradient"):
        # sprawdzić czy liczba wierszy w X=długośc wektora y
        # sprawdzić czy X nie jest macierzą zerową
        self.A = X
        self.coef_ = np.full(shape=X.shape[1], fill_value=0)
        self.b = y
        self.L = norm(X) ** 2

        if method == "gradient":
            return self.__gradient_method(y=self.coef_, L=self.L)
        elif method == "dual_gradient":
            return self.__dual_gradient_method(v=self.coef_, L=self.L)
        elif method == "accelerated":
            return self.__accelerated_method(x=self.coef_, L=self.L, coef=coef)
        else:
            print("wrong argument")
            return None
