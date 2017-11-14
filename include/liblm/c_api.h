//
// Created by Doge on 2017/11/14.
//

#ifndef LIBLM_C_API_H
#define LIBLM_C_API_H


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
#include <cstdlib>

struct dmatrix {
    double *data;
    size_t row;
    size_t col;
};

struct lm_param {
    int alg;
    int regu;
    double lambda1;
    double lambda2;
    double leanring_rate;
    size_t n_iter;
    double e;
};

struct lm_problem {
    dmatrix X;
    double *y;
};

struct lm_model {
    int type;
    double *coef;
    double *mean;
    double *var;
    size_t n_features;
};

enum { NONE, L1, L2, L1L2};
enum { CLF, REG };

lm_model* lm_train(const lm_problem *prob, const lm_param *param);
double* lm_predict(const lm_model *model, dmatrix *X);
void save_model(const char *filename, const lm_model *model);
lm_model* load_model(const char *filename);
void free_model(lm_model *model);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif //LIBLM_C_API_H
