//
// Created by Doge on 2017/11/14.
//

#ifndef LIBLM_CLASSIFIER_H
#define LIBLM_CLASSIFIER_H

#include "base.h"
#include <Eigen/Dense>
using namespace Eigen;


class Classifier :public LinearModel {
public:
    virtual VectorXd predict(const MatrixXd &X);
protected:
    VectorXd sigmoid(const MatrixXd &X);
private:
    VectorXd sign(const VectorXd &y);

};


class LogisticRegression :public Classifier {
public:
    LogisticRegression(double learning_rate, size_t n_iter, double e) :
            learning_rate_(learning_rate), n_iter_(n_iter), e_(e) {
        Classifier();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);

private:
    VectorXd diff(const MatrixXd &X, const VectorXd &y);
    double learning_rate_;
    size_t n_iter_;
    double e_;
};


class LogisticRegressionL1 :public Classifier {
public:
    LogisticRegressionL1(double learning_rate, size_t n_iter, double e, double lambda) :
            learning_rate_(learning_rate), n_iter_(n_iter), e_(e), lambda_(lambda) {
        Classifier();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);
private:
    double learning_rate_;
    size_t n_iter_;
    double e_;
    double lambda_;
};


class LogisticRegressionL2 :public Classifier {
public:
    LogisticRegressionL2(double learning_rate, size_t n_iter, double e, double lambda) :
            learning_rate_(learning_rate), n_iter_(n_iter), e_(e), lambda_(lambda) {
        Classifier();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);
private:
    VectorXd diff(const MatrixXd &X, const VectorXd &y);
    double learning_rate_;
    size_t n_iter_;
    double e_;
    double lambda_;
};

class LogisticRegressionL1L2 :public Classifier {
public:
    LogisticRegressionL1L2(double learning_rate, size_t n_iter, double e, double lambda1, double lambda2) :
            learning_rate_(learning_rate), n_iter_(n_iter), e_(e), lambda1_(lambda1), lambda2_(lambda2_) {
        Classifier();
    }
    virtual void fit(const MatrixXd &X, const VectorXd &y);

private:
    double learning_rate_;
    size_t n_iter_;
    double e_;
    double lambda1_;
    double lambda2_;
};

#endif //LIBLM_CLASSIFIER_H
