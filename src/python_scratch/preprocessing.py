class ScratchStandardScaler:
    def __init__(self):
        self.__mean = None
        self.__std = None
        self.__var = None
        self.__n_samples = None
    
    def fit(self, X):
        self.__mean = X.mean()
        self.__std = X.std(ddof=0)
        self.__std[self.__std == 0] = 1
        self.__var = X.var(ddof=0)
        self.__n_samples = len(X)
        
    def transform(self, X):
        z = (X - self.__mean) / self.__std
        return z.to_numpy()
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    @property
    def mean_(self):
        return self.__mean.to_numpy()
    
    @property
    def std_(self):
        return self.__std.to_numpy()
    
    @property
    def var_(self):
        return self.__var.to_numpy()
    
    @property
    def n_samples_(self):
        return self.__n_samples.to_numpy()            