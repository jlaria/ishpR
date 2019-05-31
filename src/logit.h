double R_logit(const arma::vec & beta_k, 
               const arma::mat & X_k,
               const arma::vec & eta_minus_k,
               const arma::mat & y){
  arma::vec eta = eta_minus_k + X_k*beta_k;
  return arma::mean(arma::log(1 + arma::exp(eta)) - y % eta);
}
double R_v_logit(const arma::vec & beta, 
               const arma::mat & X,
               const arma::mat & y){
  arma::vec eta = X*beta;
  return arma::mean(arma::log(1 + arma::exp(eta)) - y % eta);
}


arma::vec grad_R_logit(const arma::vec & beta_k, 
                       const arma::mat & X_k,
                       const arma::vec & eta_minus_k,
                       const arma::mat & y){
  
  arma::vec eta = eta_minus_k + X_k*beta_k;
  arma::vec d = 1/(1 + arma::exp(-eta)) - y;
  return X_k.t() * d/y.n_elem;
}
