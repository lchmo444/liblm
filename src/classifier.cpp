//
// Created by Doge on 2017/11/14.
//

#include <liblm/classifier.h>

VectorXd Classifier::predict(const MatrixXd &X)
{
    MatrixXd X_test = combine_bias(scalaer->transform(X));
    return sign(sigmoid(X_test));
}

VectorXd Classifier::sign(const VectorXd &y) {
    VectorXd t(y.rows());
    for (size_t i = 0; i < y.rows(); ++i)
        t(i) = y(i) >= 0.5 ? 1.0 : 0.0;
    return t;
}

VectorXd Classifier::sigmoid(const MatrixXd &X) {
    return 1.0 / (1 + (X * coef_).array().exp());
}


void LogisticRegression::fit(const MatrixXd &X, const VectorXd &y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));

    coef_ = VectorXd::Zero(X_train.cols());

    size_t i;
    for (i = 0; i < n_iter_; ++i)
    {
        VectorXd g = diff(X_train, y);
        if (g.norm() < e_)
            break;
        coef_ += learning_rate_ * g;
    }

}

VectorXd LogisticRegression::diff(const MatrixXd &X, const VectorXd &y) {
    return (X.transpose() * (sigmoid(X) - y)).matrix();
}

void LogisticRegressionL2::fit(const MatrixXd & X, const VectorXd & y)
{
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));

    coef_ = VectorXd::Zero(X_train.cols());

    size_t i;
    for (i = 0; i < n_iter_; ++i)
    {
        VectorXd g = diff(X_train, y);
        if (g.norm() < e_)
            break;
        coef_ += learning_rate_ * g;
    }

}

VectorXd LogisticRegressionL2::diff(const MatrixXd &X, const VectorXd &y) {
    return (X.transpose() * (sigmoid(X) - y)).matrix() - lambda_ * coef_;
}

void LogisticRegressionL1::fit(const MatrixXd & X, const VectorXd & y)
{
    //TODO: Add this method
}

void LogisticRegressionL1L2::fit(const MatrixXd & X, const VectorXd & y)
{
    //TODO: Add this method
}
