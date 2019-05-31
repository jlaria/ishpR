
double R_linear(const arma::vec & beta_k, 
                const arma::mat & X_k,
                const arma::vec & eta_minus_k,
                const arma::mat & y){
  arma::vec r = y - eta_minus_k - X_k*beta_k;
  return arma::sum(arma::square(r))/(2*y.n_elem);
  //return arma::as_scalar(r.t()*r)/(2*y.n_elem);
}
double R_v_linear(const arma::vec & beta, 
                const arma::mat & X,
                const arma::mat & y){
  arma::vec r = y - X*beta;
  return arma::sum(arma::square(r))/(2*y.n_elem);
  //return arma::as_scalar(r.t()*r)/(2*y.n_elem);
}

arma::vec grad_R_linear(const arma::vec & beta_k, 
                        const arma::mat & X_k,
                        const arma::vec & eta_minus_k,
                        const arma::mat & y){
  return -X_k.t()*(y - eta_minus_k - X_k*beta_k)/y.n_elem;
}
