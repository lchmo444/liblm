#include <iostream>
#include <cstdio>
#include <Eigen/Dense>

#include "src/liblm/c_api.h"
#include "src/liblm/liblm.h"

using namespace Eigen;

void test_c() {

    double *X = (double *) malloc(sizeof(double) * 5);
    double *y = (double *) malloc(sizeof(double) * 5);
    double *X_test = (double *) malloc(sizeof(double) * 5);
    for (int i = 0; i < 5; ++i) {
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
}


void test_cpp() {
    MatrixXd X(4, 1);
    X << 1, 2, 3, 4;
    VectorXd y(4);
    y << 1, 2, 3, 4;
    MatrixXd X_test = X.array() + 5;
    Regressor *clf = new LinearRegression();
    clf->fit(X, y);
    std::cout << clf->predict(X_test) << std::endl;
    delete clf;
}

int main() {
    test_c();
    std::cout << "----------------" << std::endl;
    test_cpp();
    return 0;
}