//
// Created by Doge on 2017/11/14.
//

#include "c_api.h"
#include "base.h"
#include "classifier.h"
#include "regressor.h"
#include "preprocess.h"

#include <cstdlib>
#include <cstdio>

using namespace std;

lm_model* lm_train(const lm_problem *prob, const lm_param *param) {
    LinearModel *lmodel;

    if (param->alg == REG)
    {
        if (param->regu == NONE)
            lmodel = new LinearRegression();
        else if (param->regu == L1)
            lmodel = new Lasso(param->lambda1, param->n_iter, param->e);
        else if (param->regu == L2)
            lmodel = new Ridge(param->lambda2);
        else if (param->regu == L1L2)
            lmodel = new ElasticNet(param->lambda1, param->lambda2, param->n_iter, param->e);
        else
        {
            printf("Please choose correct regularization method!\n");
            return NULL;
        }
    }
    else if (param->alg == CLF)
    {
        if (param->regu == NONE)
            lmodel = new LogisticRegression(param->leanring_rate, param->n_iter, param->e);
        else if (param->regu == L1)
            lmodel = new LogisticRegressionL1(param->leanring_rate, param->n_iter, param->e, param->lambda1);
        else if (param->regu == L2)
            lmodel = new LogisticRegressionL2(param->leanring_rate, param->n_iter, param->e, param->lambda2);
        else if (param->regu == L1L2)
            lmodel = new LogisticRegressionL1L2(param->leanring_rate, param->n_iter, param->e, param->lambda1, param->lambda2);
        else
        {
            printf("Please choose correct regularization method!\n");
            return NULL;
        }
    }
    else
    {
        printf("Please choose correct algorithm!\n");
        return NULL;
    }

    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X(prob->X.data, prob->X.row, prob->X.col);
    Map<VectorXd> y(prob->y, prob->X.row);
    lmodel->fit(X, y);
    LModel m = lmodel->get_model();
    size_t n_features = m.coef.rows();

    double *coef = (double *)malloc(sizeof(double) * n_features);
    memcpy(coef, m.coef.data(), sizeof(double) * n_features);
    double *mean = (double *)malloc(sizeof(double) * n_features);
    memcpy(mean, m.mean.data(), sizeof(double) * n_features);
    double *var = (double *)malloc(sizeof(double) * n_features);
    memcpy(var, m.var.data(), sizeof(double) * n_features);

    lm_model *model = (lm_model*)malloc(sizeof(lm_model));
    if (param->alg == CLF)
        model->type = CLF;
    else
        model->type = REG;
    model->n_features = n_features;
    model->coef = coef;
    model->mean = mean;
    model->var = var;
    delete lmodel;
    return model;
}

void free_model(lm_model *model) {
    free(model->coef);
    free(model->mean);
    free(model->var);
    free(model);
}

double* lm_predict(const lm_model *model, dmatrix *X) {
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> X_test(X->data, X->row, X->col);
    LinearModel *lm;
    if (model->type == CLF)
        lm = new Classifier();
    else
        lm = new Regressor();
    Map<VectorXd> coef(model->coef, model->n_features);
    Map<RowVectorXd> mean(model->mean, model->n_features);
    Map<RowVectorXd> var(model->var, model->n_features);
    lm->set_model(coef, mean, var);

    VectorXd ypred = lm->predict(X_test);
    double *y = (double *)malloc(sizeof(double) * ypred.rows());
    memcpy(y, ypred.data(), sizeof(double) * ypred.rows());
    delete lm;

    return y;
}

void save_model(const char *filename, const lm_model *model) {
    FILE *fp = fopen(filename, "wb");
    fwrite(&model->n_features, sizeof(size_t), 1, fp);
    fwrite(&model->type, sizeof(int), 1, fp);
    fwrite(model->coef, sizeof(double), model->n_features, fp);
    fwrite(model->mean, sizeof(double), model->n_features, fp);
    fwrite(model->var, sizeof(double), model->n_features, fp);
    fclose(fp);
}

lm_model* load_model(const char *filename) {
    lm_model *model = (lm_model*)malloc(sizeof(lm_model));
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("File `%s` does not exist!\n", filename);
        return NULL;
    }
    fread(&model->n_features, sizeof(size_t), 1, fp);
    fread(&model->type, sizeof(int), 1, fp);
    model->coef = (double *)malloc(sizeof(double) * model->n_features);
    model->mean = (double *)malloc(sizeof(double) * model->n_features);
    model->var = (double *)malloc(sizeof(double) * model->n_features);

    fread(model->coef, sizeof(double), model->n_features, fp);
    fread(model->mean, sizeof(double), model->n_features, fp);
    fread(model->var, sizeof(double), model->n_features, fp);
    fclose(fp);
    return model;
}
