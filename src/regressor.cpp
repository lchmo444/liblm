//
// Created by Doge on 2017/11/14.
//

#include <liblm/regressor.h>

VectorXd Regressor::predict(const MatrixXd &X) {
    MatrixXd X_test = combine_bias(scalaer->transform(X));
    return X_test * coef_;
}

void LinearRegression::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = (X_train.transpose() * X_train).inverse()*X_train.transpose()*y;
}

void Lasso::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = VectorXd::Zero(X_train.cols());
    for (size_t iter = 0; iter < n_iter_; ++iter) {
        RowVectorXd z = (X_train.array() * X_train.array()).colwise().sum();
        VectorXd tmp = VectorXd::Zero(X_train.cols());
        assert(z.rows() == tmp.cols());
        for (size_t k = 0; k < X_train.cols(); ++k) {
            double wk = coef_(k);
            coef_(k) = 0;
            double p_k = X_train.col(k).transpose() * (y - X_train * coef_);
            double w_k = 0.0;
            if (p_k < -lambda_ / 2.0)
                w_k = (p_k + lambda_ / 2.0) / z(k);
            else if (p_k > lambda_ / 2.0)
                w_k = (p_k - lambda_ / 2.0) / z(k);
            else
                w_k = 0.0;
            tmp(k) = w_k;
            coef_(k) = wk;
        }
        if ((coef_ - tmp).norm() < e_)
            break;
        coef_ = tmp;
    }
}

void Ridge::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = (X_train.transpose() * X_train + lambda_
                                             * MatrixXd::Identity(X_train.cols(), X_train.cols())).inverse()*X_train.transpose()*y;
}

void ElasticNet::fit(const MatrixXd & X, const VectorXd & y) {
    MatrixXd X_train = combine_bias(scalaer->fit_transform(X));
    coef_ = VectorXd::Zero(X_train.cols());
    for (size_t iter = 0; iter < n_iter_; ++iter) {
        RowVectorXd z = (X_train.array() * X_train.array()).colwise().sum() + lambda2_;
        VectorXd tmp = VectorXd::Zero(X_train.cols());
        assert(z.rows() == tmp.cols());
        for (size_t k = 0; k < X_train.cols(); ++k) {
            double wk = coef_(k);
            coef_(k) = 0;
            double p_k = X_train.col(k).transpose() * (y - X_train * coef_);
            double w_k = 0.0;
            if (p_k < -lambda1_ / 2.0)
                w_k = (p_k + lambda1_ / 2.0) / z(k);
            else if (p_k > lambda1_ / 2.0)
                w_k = (p_k - lambda1_ / 2.0) / z(k);
            else
                w_k = 0.0;
            tmp(k) = w_k;
            coef_(k) = wk;
        }
        if ((coef_ - tmp).norm() < e_)
            break;
        coef_ = tmp;
    }
}