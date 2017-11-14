//
// Created by Doge on 2017/11/14.
//

#include <liblm/base.h>


VectorXd LinearModel::predict(const MatrixXd & X) {
    MatrixXd X_test = combine_bias(scalaer->transform(X));
    return X_test * coef_;
}

MatrixXd LinearModel::combine_bias(const MatrixXd & X) {
    MatrixXd t(X.rows(), X.cols() + 1);
    t.col(X.cols()) = VectorXd::Ones(X.rows());
    t.leftCols(X.cols()) = X;
    return t;
}