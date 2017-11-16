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
### C++
```c++
#include <liblm/classifier.h>
#include <iostream>

using namespace std;
int main()
{
    //TODO: ADD
    return 0;
}
```
