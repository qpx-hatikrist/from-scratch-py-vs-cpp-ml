class ScratchLinearRegression:
    def __init__(self, max_iter=100, solver='analytic'):
        if isinstance(max_iter, int) and not isinstance(max_iter, bool):
            if max_iter < 1:
                raise ValueError('max_iter must be positive')
            self.max_iter = max_iter
        else:
            raise TypeError('max_iter need to be int type')

        if solver == 'analytic':
            self.solver = solver
        else:
            raise TypeError('solver can be "analytic"')
        self.coef_ = None
        
        
    def fit(self, X_train, y_train):
        #self.coef_, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
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