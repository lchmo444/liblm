//
// Created by Doge on 2017/11/14.
//

#include "../liblm/preprocess.h"


void StandardScaler::fit(const MatrixXd & X) {
    mean_ = X.colwise().mean();
    auto t = X.rowwise() - mean_;
    var_ = (t.array() * t.array()).matrix().colwise().sum().array().sqrt();
}

MatrixXd StandardScaler::transform(const MatrixXd & X) {
    MatrixXd t(X);
    for (int i = 0; i < X.cols(); ++i)
        t.col(i) /= var_(i);
    return t;
}

MatrixXd StandardScaler::fit_transform(const MatrixXd & X) {
    fit(X);
    return transform(X);
}