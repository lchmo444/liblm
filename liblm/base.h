//
// Created by Doge on 2017/11/14.
//

#ifndef LIBLM_BASE_H
#define LIBLM_BASE_H

#include "preprocess.h"
#include <Eigen\Dense>

using namespace Eigen;


struct LModel {
    RowVectorXd mean;
    RowVectorXd var;
    VectorXd coef;
};


class LinearModel {
public:
    LinearModel() {
        scalaer = new StandardScaler();
    }
    ~LinearModel() {
        delete scalaer;
    }

    virtual void fit(const MatrixXd &X, const VectorXd &y) {};
    virtual VectorXd predict(const MatrixXd &X);

    void set_model(const VectorXd &coef, const RowVectorXd &mean, const RowVectorXd &var) {
        coef_ = coef;
        scalaer->set_param(mean, var);
    }
    LModel get_model() {
        LModel model;
        model.coef = coef_;
        model.mean = scalaer->get_mean();
        model.var = scalaer->get_var();
        return model;
    }

protected:
    MatrixXd combine_bias(const MatrixXd &X);
    StandardScaler *scalaer;
    VectorXd coef_;
};





#endif //LIBLM_BASE_H
