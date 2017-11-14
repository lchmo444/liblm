//
// Created by Doge on 2017/11/14.
//

#ifndef LIBLM_REGRESSOR_H
#define LIBLM_REGRESSOR_H

#include "base.h"
#include <Eigen/Dense>
using namespace Eigen;

class Regressor :public LinearModel {
public:
    virtual VectorXd predict(const MatrixXd &X);
};


class LinearRegression :public Regressor {
public:
    virtual void fit(const MatrixXd &X, const VectorXd &y);
};


class Lasso :public Regressor {
public:
    Lasso(double lambda, size_t n_iter, double e) :
            lambda_(lambda), n_iter_(n_iter), e_(e) {
        Regressor();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);
private:
    size_t n_iter_;
    double e_;
    double lambda_;
};



class Ridge :public Regressor {
public:
    Ridge(double lambda) :lambda_(lambda) { Regressor(); }
    virtual void fit(const MatrixXd &X, const VectorXd &y);
private:
    double lambda_;
};



class ElasticNet :public Regressor {
public:
    ElasticNet(double lambda1, double lambda2, size_t n_iter, double e) :
            lambda1_(lambda1), lambda2_(lambda2_), n_iter_(n_iter), e_(e) {
        Regressor();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);
private:
    double lambda1_;
    double lambda2_;
    size_t n_iter_;
    double e_;
};



#endif //LIBLM_REGRESSOR_H
