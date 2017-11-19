# liblm
A linear model library by C++(with C interface).
including:
* Linear Regression
* Lasso
* Ridge
* Elastic Net
* Logistic Regression
## Usage
### Python
You can use liblm like scikit-learn.
```python
import numpy as np
import liblm

X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([1, 2, 3, 4])
X_test = X + 5

clf = liblm.Lasso(lamda=0.2)
clf.fit(X, y)
print(clf.predict(X_test))
```
### C
```c
#include <stdio.h>
#include <liblm/c_api.h>
  
int main() {
    double *X = (double *) malloc(sizeof(double) * 5);
    double *y = (double *) malloc(sizeof(double) * 5);
    double *X_test = (double *) malloc(sizeof(double) * 5);
    for (int i = 0; i < 5; ++i) 
    {
        X[i] = i;
        y[i] = 2 * i - 1;
        X_test[i] = i + 5;
    }
    lm_problem *prob = (lm_problem *) malloc(sizeof(lm_problem));
    dmatrix data;
    data.data = X;
    data.row = 5;
    data.col = 1;
    prob->X = data;
    prob->y = y;
    lm_param param;
    param.alg = REG;
    param.regu = NONE;
    lm_model *model = lm_train(prob, &param);
    dmatrix d;
    d.data = X_test;
    d.row = 5;
    d.col = 1;
    y = lm_predict(model, &d);
    for (int i = 0; i < 5; ++i)
        printf_s("%.3f ", y[i]);
    printf_s("\n");
    free_model(model);
    return 0;
}

```

### C++
```c++
#include <liblm/regressor.h>
#include <Eigen/Dense>
#include <iostream>
  
using namespace Eigen;
  
int main()
{
    MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    VectorXd y(4);
    y << 1, 2, 3, 4;
    MatrixXd X_test = X.array() + 5;
    Regressor *clf = new LinearRegression();
    clf->fit(X, y);
    std::cout << clf->predict(X_test) << std::endl;
    delete clf;
    return 0;
}
```
## Build dynamic link library
### For Windows
Use cmake GUI to generate *Visual Studio* project, and build it by VS.
### For Linux or Mac
* `$ cmake .`
* `$ make`

After building, you will get a library file and a executable file.

## Build Python-package
Put library file(`liblm.dll` or `libliblm.so` or `libliblm.dylib`) into `python/liblm`.  
* `$ cd python`
* `$ python setup.py install`