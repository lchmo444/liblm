#include <iostream>
#include <liblm/c_api.h>

using namespace std;

void test() {

    double *X = (double *)malloc(sizeof(double) * 5);
    double *y = (double *)malloc(sizeof(double) * 5);
    double *X_test = (double *)malloc(sizeof(double) * 5);
    for (int i = 0; i < 5; ++i)
    {
        X[i] = i;
        y[i] = 2 * i - 1;
        X_test[i] = i + 5;
    }
    lm_problem *prob = (lm_problem *)malloc(sizeof(lm_problem));
    dmatrix data;
    data.data = X;
    data.row = 5;
    data.col = 1;
    prob->X = data;
    prob->y = y;
    lm_param param;
    param.alg = REG;
    param.regu = NONE;
    lm_model*model = lm_train(prob, &param);
    dmatrix d;
    d.data = X_test;
    d.row = 5;
    d.col = 1;
    y = lm_predict(model, &d);
    for (int i = 0; i < 5; ++i)
        cout << y[i] << " ";
    cout << endl;
}

int main() {
    test();
    return 0;
}