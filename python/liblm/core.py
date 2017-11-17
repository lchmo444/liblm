import numpy as np
from ctypes import *
import sys

__all__ = ['LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'LogisticRegression']

# Load C-module.
if sys.platform == 'win32':
    _lib = CDLL("./liblm.dll")
elif sys.platform == 'darwin':
    _lib = CDLL("./libliblm.dylib")
else:
    _lib = CDLL("./libliblm.so")


def fillprototype(f, restype, argtypes):
    """A helper function to decorate C-function
    Parameters
    ----------
    f: object C-function
    restype: list, return value type
    argtypes: list, arguments type

    Examples
    --------
    double add(double x, double y)
    {
        return x + y;    
    }
    fillprototype(add, [c_double, c_double])
    
    """
    f.restype = restype
    f.argtypes = argtypes

def to_cstr(filename):
    """Transform type `str` to `bytes`."""
    return filename.encode("utf-8") if isinstance(filename, str) else filename


# Enum type
(NONE, L1, L2, L1L2) = (0, 1, 2, 3)
(CLF, REG) = (0, 1)


# Some C-struct.
class dmatrix(Structure):
    _fields_ = [("data", POINTER(c_double)), 
                ("row", c_size_t),
                ("col", c_size_t)]


class lm_param(Structure):
    _fields_ = [("alg", c_int), 
                ("regu", c_int),
                ("lambda1", c_double),
                ("lambda2", c_double),
                ("learning_rate", c_double),
                ("n_iter", c_size_t),
                ("e", c_double)]


class lm_problem(Structure):
    _fields_ = [("X", dmatrix), 
                ("y",POINTER(c_double))]
        

class lm_model(Structure):
    _fields_ = [("type", c_int), 
                ("coef", POINTER(c_double)),
                ("mean", POINTER(c_double)),
                ("var", POINTER(c_double)),
                ("n_features", c_size_t)]


fillprototype(_lib.lm_train, POINTER(lm_model), [POINTER(lm_problem), POINTER(lm_param)])
fillprototype(_lib.lm_predict, POINTER(c_double), [POINTER(lm_model), POINTER(dmatrix)])
fillprototype(_lib.save_model, None, [c_char_p, POINTER(lm_model)])
fillprototype(_lib.load_model, POINTER(lm_model), [c_char_p])
fillprototype(_lib.free_model, None, [POINTER(lm_model)])


class LinearModel(object):

    def __init__(self, alg, regu, lambda1, lambda2, learning_rate, n_iter, e):
        """Initialize the model parameters.

        Parameters
        ----------
        alg: `CLF` or REG``.
        regu: Regularization type, `NONE`, `L1`. `L2`, or `L1L2`.
        lambda1: L1 regularization arg.
        lambda2: L2 regularization arg.
        learning_rate: learning_rate.
        n_iter: iteration rounds.
        e: error rate, if ||Wn+1 - Wn|| <= e, then stop iteration.

        """
        self._param = lm_param()
        self._param.alg = alg
        self._param.regu = regu
        self._param.lambda1 = lambda1
        self._param.lambda2 = lambda2
        self._param.learning_rate = learning_rate
        self._param.n_iter = n_iter
        self._param.e = e

    def fit(self, X, y):
        """Fit linear model by transfer C-function.
        Parameters
        ----------
        X: np.ndarray. shape=(n_samples, n_features).
        y: np.ndarray. shape=(n_samples, ).

        Returns
        -------
        self.

        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be np.ndarray")
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be np.ndarray")
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        if y.dtype != np.float64:
            y = y.astype(np.float64)

        data = X.ctypes.data_as(POINTER(c_double))
        y = y.ctypes.data_as(POINTER(c_double))
        dmat = dmatrix()
        dmat.data = data

        dmat.row = X.shape[0]
        dmat.col = X.shape[1]
        prob = lm_problem()
        prob.X = dmat
        prob.y = y
        self._model = _lib.lm_train(byref(prob), byref(self._param))
        return self

    def predict(self, X):
        """Predict `X` by transfer C-function.
        Parameters
        ----------
        X: np.ndarray. shape=(n_samples, n_features).

        Returns
        -------
        ypred, np.ndarray. shape=(n_samples, ).

        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be np.ndarray")
        if X.dtype != np.float64:
            X = X.astype(np.float64)

        data = X.ctypes.data_as(POINTER(c_double))
        dmat = dmatrix()
        dmat.data = data
        dmat.row = X.shape[0]
        dmat.col = X.shape[1]
        c_y_arr = _lib.lm_predict(self._model, dmat)
        return np.ctypeslib.as_array(c_y_arr, shape=(X.shape[0], ))


    def load_model(self, filename):
        """Load model from disk.
        Parameters
        ----------
        filename: model's file name.
        """
        self._model = _lib.load_model(to_cstr(filename))

    def save_model(self, filename):
        """Save model to disk.
        Parameters
        ----------
        filename: model's file name.
        """
        _lib.save_model(to_cstr(filename), self._model)


    def __del__(self):
        if hasattr(self, '_model'):
            _lib.free_model(self._model)





class LinearRegression(LinearModel):
    def __init__(self):
        super().__init__(REG, NONE, 0, 0, 0, 0, 0)

class Lasso(LinearModel):
    def __init__(self, lamda=0.2, n_iter=100, e=0.01):
        super().__init__(REG, L1, lamda, 0, 0, n_iter, e)

class Ridge(LinearModel):
    def __init__(self, lamda=0.2):
        super().__init__(REG, L2, 0, lamda, 0, 0, 0)

class ElasticNet(LinearModel):
    def __init__(self, lamda1=0.2, lamda2=0.2, n_iter=100, e=0.01):
        super().__init__(REG, L1L2, lamda1, lamda2, 0, n_iter, e)


class LogisticRegression(LinearModel):
    def __init__(self, learning_rate=0.05, n_iter=100, e=0.01, **kwargs):
        super().__init__(CLF, NONE, 0, 0, learning_rate, n_iter, e)


if __name__ == '__main__':
    X = np.array([[0], [1], [2], [3], [4]])
    X_test = X + 5
    y = 2 * X.flatten() - 1
    reg = Lasso()
    reg.fit(X, y)
    # reg.save_model("aa.model")
    print(reg.predict(X_test))
    reg.save_model("bb.model")
    del reg
    reg = Lasso()
    reg.load_model("bb.model")
    print(reg.predict(X_test))