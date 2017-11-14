//
// Created by Doge on 2017/11/14.
//

#ifndef LIBLM_PREPROCESS_H
#define LIBLM_PREPROCESS_H

#include <Eigen/Dense>
using namespace Eigen;

class StandardScaler {
public:
    void fit(const MatrixXd &X);
    MatrixXd transform(const MatrixXd &X);
    MatrixXd fit_transform(const MatrixXd &X);
    void set_param(const RowVectorXd &mean, const RowVectorXd &var)
    {
        mean_ = mean;
        var_ = var;
    }
    RowVectorXd get_mean() {
        return mean_;
    }
    RowVectorXd get_var() {
        return var_;
    }
private:
    RowVectorXd mean_;
    RowVectorXd var_;
};


#endif //LIBLM_PREPROCESS_H
