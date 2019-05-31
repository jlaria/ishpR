#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]
#include <algorithm>

const int TYPE_LINEAR = 0;
const int TYPE_LOGIT = 1;
const int TYPE_COX = 2;

const int MAX_ITER = 20;
const int MAX_ITER_INNER = 200;

const int LOSS_MSE = 0;
const int LOSS_LOGISTIC = 1;
const int LOSS_COX = 2;

const int REGULARIZATION_SGL = 0;
const int REGULARIZATION_LASSO = 1;
const int REGULARIZATION_GLASSO = 2;

#include "optimization.h"
#include "linear.h"
#include "logit.h"
#include "coxph.h"

// [[Rcpp::export]]
Rcpp::List isglasso(
    const arma::mat & X_t, 
    const arma::mat & y_t,
    const arma::mat & X_v, 
    const arma::mat & y_v,
    const arma::vec & grp_len,
    const int num_iter,
    const int loss,
    const int metric,
    const int regularization
){
  
  double (*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &);
  double (*R_v)(const arma::vec &, const arma::mat &, const arma::mat &);
  arma::vec (*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &);
  double t0 = 1;
  
  switch (loss){
  case LOSS_MSE:
    R = R_linear;
    grad_R = grad_R_linear;
    break;
  case LOSS_LOGISTIC:
    R = R_logit;
    grad_R = grad_R_logit;
    break;
  case LOSS_COX:
    R = R_coxph;
    grad_R = grad_R_coxph;
    t0 = 0.2;
    break;
  default:
    R = R_linear;
    grad_R = grad_R_linear;
    break;  
  }
  
  switch (metric){
  case LOSS_MSE:
    R_v = R_v_linear;
    break;
  case LOSS_LOGISTIC:
    R_v = R_v_logit;
    break;
  case LOSS_COX:
    R_v = R_v_coxph;
    break;
  default:
    R_v = R_v_linear;
    break;
  }
  
  Rcpp::List fit;
  
  switch (regularization){
  case REGULARIZATION_LASSO:
    fit = il_rs(R, R_v, grad_R, X_t, y_t, X_v, y_v, num_iter, t0);
    break;
  case REGULARIZATION_GLASSO:
    fit = igl_rs(R, R_v, grad_R, X_t, y_t, X_v, y_v, grp_len, num_iter, t0);
    break;
  default:
    fit = isgl_rs(R, R_v, grad_R, X_t, y_t, X_v, y_v, grp_len, num_iter, t0);
    break;
  }
  
  return fit;
}


