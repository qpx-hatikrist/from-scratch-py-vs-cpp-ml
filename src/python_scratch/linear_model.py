import time
import pandas as pd
import numpy as np
from tqdm import tqdm

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6} seconds")
        return result
    return wrapper

class ScratchLinearRegression:
    def __init__(self):
        self.coef_ = None
        
    @timeit    
    def fit(self, X_train, y_train):
        
        # приведение к numpy, чтобы не было KeyError из pandas
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = np.array(y_train)
            
        Xb = self._add_intercept(X_train)
        Xt = self._transpose(Xb)
        XtXb = self._matmul(Xt, Xb)
        Xty = self._matvec(Xt, y_train)
        
        # маленький ridge-костыль
        lambda_ = 1e-6
        n = len(XtXb)              # размер квадратной матрицы
        XtX_reg = []
        for i in range(n):
            row = XtXb[i][:]       # копия строки
            if i != 0:          # не штрафуем bias
                row[i] += lambda_     # добавляем λ к диагональному элементу
            XtX_reg.append(row)

        self.coef_ = self._solve_linear_system(XtX_reg, Xty)
            
        return self.coef_
    @timeit    
    def predict(self, X_test):
        if self.coef_ is None:
            raise ValueError('model is not fitted')
        Xb = self._add_intercept(X_test)
        return self._matvec(Xb, self.coef_)
    
    @staticmethod
    def _add_intercept(A):
        return [[1.0] + list(row) for row in A]
        
    @staticmethod
    def _transpose(A):
        # A: m x n -> n x m
        return [list(row) for row in zip(*A)]
    
    @staticmethod
    def _matmul(A, B):
        # A: m x n, B: n x k -> m x k
        m, n = len(A), len(A[0])
        n2, k = len(B), len(B[0])
        assert n == n2
        C = [[0.0 for _ in range(k)] for _ in range(m)]
        for i in range(m):
            for j in range(k):
                s = 0.0
                for t in range(n):
                    s += A[i][t] * B[t][j]
                C[i][j] = s
        return C
    
    @staticmethod
    def _matvec(A, v):
        # A: m x n, v: n -> m
        m, n = len(A), len(A[0])
        assert len(v) == n
        res = [0.0 for _ in range(m)]
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i][j] * v[j]
            res[i] = s
        return res
    
    @staticmethod
    def _solve_linear_system(A, b):
        """
        Решаем систему A w = b методом Гаусса.
        A — квадратная матрица n x n (list[list[float]])
        b — вектор длины n (list[float])
        """
        n = len(A)

        # 1. Делаем копию A и формируем расширенную матрицу [A | b]
        M = []
        for i in range(n):
            row = A[i][:]        # копия строки A[i]
            row.append(b[i])     # добавляем соответствующий элемент b
            M.append(row)

        # 2. Прямой ход: делаем матрицу верхнетреугольной
        for i in range(n):
            # ищем строку с максимальным по модулю элементом в столбце i (для устойчивости)
            max_row = i
            for r in range(i + 1, n):
                if abs(M[r][i]) > abs(M[max_row][i]):
                    max_row = r

            # если опорный элемент почти ноль — считаем, что матрица вырожденная
            if abs(M[max_row][i]) < 1e-12:
                raise ValueError("Matrix is singular or nearly singular")

            # меняем текущую строку с найденной
            M[i], M[max_row] = M[max_row], M[i]

            # нормализуем строку: делаем опорный элемент равным 1
            pivot = M[i][i]
            for c in range(i, n + 1):
                M[i][c] /= pivot

            # обнуляем элементы ниже в этом столбце
            for r in range(i + 1, n):
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]

        # 3. Обратный ход: находим w снизу вверх
        w = [0.0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            s = M[i][n]  # свободный член (последний элемент строки)
            for j in range(i + 1, n):
                s -= M[i][j] * w[j]
            w[i] = s

        return w

'--------------------------------------------------RIDGE--------------------------------------------------'

class ScratchRidge:
    def __init__(self, alpha=1):
        self.coef_ = None
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        
        
    def fit(self, X_train, y_train):
        if isinstance(X_train, (pd.DataFrame, pd.Series)):
            X_train = X_train.to_numpy()
        if isinstance(y_train, (pd.Series, pd.DataFrame)):
            y_train = np.array(y_train)
        Xb = self._add_intercept(X_train)
        Xt = self._transpose(Xb)
        XtXb = self._matmul(Xt, Xb)
        Xty = self._matvec(Xt, y_train)
        
        n = len(XtXb)
        XtX_reg = []
        for i in range(n):
            row = XtXb[i][:]
            if i != 0:                      # не штрафуем bias
                row[i] += self.alpha
            XtX_reg.append(row)

        self.coef_ = self._solve_linear_system(XtX_reg, Xty)
            
        return self.coef_
        
    def predict(self, X_test):
        if self.coef_ is None:
            raise ValueError('model is not fitted')
        Xb = self._add_intercept(X_test)
        return self._matvec(Xb, self.coef_)
    
    @staticmethod
    def _add_intercept(A):
        return [[1.0] + list(row) for row in A]
        
    @staticmethod
    def _transpose(A):
        # A: m x n -> n x m
        return [list(row) for row in zip(*A)]
    
    @staticmethod
    def _matmul(A, B):
        # A: m x n, B: n x k -> m x k
        m, n = len(A), len(A[0])
        n2, k = len(B), len(B[0])
        assert n == n2
        C = [[0.0 for _ in range(k)] for _ in range(m)]
        for i in range(m):
            for j in range(k):
                s = 0.0
                for t in range(n):
                    s += A[i][t] * B[t][j]
                C[i][j] = s
        return C
    
    @staticmethod
    def _matvec(A, v):
        # A: m x n, v: n -> m
        m, n = len(A), len(A[0])
        assert len(v) == n
        res = [0.0 for _ in range(m)]
        for i in range(m):
            s = 0.0
            for j in range(n):
                s += A[i][j] * v[j]
            res[i] = s
        return res
    
    @staticmethod
    def _solve_linear_system(A, b):
        """
        Решаем систему A w = b методом Гаусса.
        A — квадратная матрица n x n (list[list[float]])
        b — вектор длины n (list[float])
        """
        n = len(A)

        M = []
        for i in range(n):
            row = A[i][:]
            row.append(b[i])
            M.append(row)

        for i in range(n):
            max_row = i
            for r in range(i + 1, n):
                if abs(M[r][i]) > abs(M[max_row][i]):
                    max_row = r

            if abs(M[max_row][i]) < 1e-12:
                raise ValueError("Matrix is singular or nearly singular")

            M[i], M[max_row] = M[max_row], M[i]

            pivot = M[i][i]
            for c in range(i, n + 1):
                M[i][c] /= pivot

            for r in range(i + 1, n):
                factor = M[r][i]
                for c in range(i, n + 1):
                    M[r][c] -= factor * M[i][c]

        w = [0.0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            s = M[i][n]
            for j in range(i + 1, n):
                s -= M[i][j] * w[j]
            w[i] = s

        return w
    
'--------------------------------------------------LASSO--------------------------------------------------'
class ScratchLasso:
    def __init__(self, alpha=1.0, max_iter=100, tol=1e-4):
        self._coef = None
        self._intercept = None
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X, y):
        n = len(X)
        p = len(X[0])
        self.coef = [0.0] * p
        X_mean = [0.0] * p

        for j in range(p):
            s = 0.0
            for i in range(n):
                s += X[i][j]
            X_mean[j] = s / n

        y_mean = sum(y_i for y_i in y) / n

        Xc = [[0.0] * p for _ in range(n)]
        for i in range(n):
            for j in range(p):
                Xc[i][j] = X[i][j] - X_mean[j]

        yc = [float(y_i - y_mean) for y_i in y]

        xx_sum = [0.0] * p
        
        for i in range(n):
            for j in range(p):
                xx_sum[j] += Xc[i][j] ** 2

        residual = [float(y_i) for y_i in yc]

        alpha_eff = self.alpha * n

        for _ in tqdm(range(self.max_iter)):
            max_delta = 0.0

            for j in range(p):
                if xx_sum[j] == 0:
                    continue
                
                rho_j = 0.0
                for i in range(n):
                    rho_j += Xc[i][j] * (residual[i] + Xc[i][j] * self.coef[j])

                # sign (soft-threshold)
                if rho_j < -alpha_eff:
                    w_new = (rho_j + alpha_eff) / xx_sum[j]
                elif rho_j > alpha_eff:
                    w_new = (rho_j - alpha_eff) / xx_sum[j]
                else:
                    w_new = 0.0

                delta = w_new - self.coef[j]
                if delta != 0.0:

                    for i in range(n):
                        residual[i] -= Xc[i][j] * delta
                    self.coef[j] = w_new
                    if abs(delta) > max_delta:
                        max_delta = abs(delta)

            if max_delta < self.tol:
                break

        self._coef = self.coef[:]
        self._intercept = y_mean - sum(X_mean[j] * self._coef[j]
                                       for j in range(p))

        return self
    
    def predict(self, X):
        y_pred = []
        for row in X:
            s = self._intercept
            for x_j, w_j in zip(row, self._coef):
                s += x_j * w_j
            y_pred.append(s)
        return y_pred         