arma::vec soft_thresh(const arma::vec & z, double l){
  arma::vec S = arma::vec(z.n_elem);
  for(int i=0; i<z.n_elem; i++){
    if(fabs(z(i)) <= l){
      S(i) = 0;
    }else{
      if(z(i)<=0){
        S(i) = z(i) + l;
      }else{
        S(i) = z(i) - l;
      }
    }
  }
  return S;
}

arma::vec update_lasso(const arma::vec & beta0, 
                       const arma::vec & gradient_R_beta0,
                       double t, 
                       double lambda1,
                       double lambda2,
                       double gammak){
  arma::vec beta1 = soft_thresh(beta0 - t*gradient_R_beta0, t*lambda1);
  return beta1;
}

arma::vec update_sparse_group_lasso(const arma::vec & beta0, 
                                    const arma::vec & gradient_R_beta0,
                                    double t, 
                                    double lambda1,
                                    double lambda2,
                                    double gammak){
  arma::vec S1 = soft_thresh(beta0 - t*gradient_R_beta0, t*lambda1);
  return std::max(1 - lambda2*gammak*t/arma::norm(S1, 2), 0.0)*S1;
}

arma::vec optimization_internal_simon_nesterov(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    arma::vec(*Update)(const arma::vec &, const arma::vec &, double, double,double,double),
    const arma::vec & beta_0,
    double lambda_1, 
    double lambda_2,
    double gamma_k,
    const arma::mat & X_k,
    const arma::vec & eta_minus_k,
    const arma::mat & y,
    double t0
){
  double p_k = X_k.n_cols;
  double N = X_k.n_rows;
  double t;
  double l = 1;
  double R_beta = 0;
  
  arma::vec beta = beta_0;
  arma::vec theta_new = beta_0;
  arma::vec theta_old = arma::zeros(p_k);
  arma::vec g = arma::zeros(p_k);
  
  // X_k.print("X_k:");
  // y.print("y:");
  // eta_minus_k.print("eta_minus_k:");
  // beta_0.print("beta_0:");
  
  do{
    // Update theta_old
    theta_old = theta_new;
    // theta_old.print("theta_old:");
    
    
    // Compute the gradient in beta0
    g = grad_R(beta, X_k, eta_minus_k, y);
    //g = g/arma::abs(g).max();
    //g.print("g:");
    
    // Compute the error in beta0
    R_beta = R(beta, X_k, eta_minus_k, y);
    
    //arma::vec r(1);
    //r.fill(R_beta);
    //r.print("r=");
    //sleep(1);
    
    //Find t such that R_updated <= R + t(g)*(beta-beta_updated) + 1/2t||beta-beta_updated||_2^2
    t = t0;
    theta_new = Update(beta, g, t, lambda_1, lambda_2, gamma_k);
    while ( arma::as_scalar(R(theta_new, X_k, eta_minus_k, y)) > 
              arma::as_scalar(R_beta + g.t()*(beta - theta_new) + 1/(2*t)*arma::sum(arma::square(beta-theta_new)))){
      t = 0.9*t;
      theta_new = Update(beta, g, t, lambda_1, lambda_2, gamma_k);
      // theta_new.print("search t theta_new:");
    }
    //theta_new.print("theta_new:");
    //theta_old.print("theta_old:");
    //beta.print("beta:");
    
    // Update beta
    beta = theta_old + l/(l+3)*(theta_new-theta_old);
    //beta.print("beta:");
    
    l++;
    
  }while(!arma::approx_equal(theta_new, theta_old, "absdiff", 0.001) && l < MAX_ITER_INNER);
  
  return beta;
}


arma::vec optimization_external_simon_nesterov(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    arma::vec(*Update)(const arma::vec &, const arma::vec &, double, double,double,double),
    const arma::vec & beta_0,
    double lambda_1, 
    double lambda_2,
    const arma::vec & gamma,
    const arma::mat & X,
    const arma::mat & y,
    const arma::vec & grp_len,
    double t0)
{
  
  int J = grp_len.n_elem;
  int p = X.n_cols;
  int N = X.n_rows;
  
  arma::vec grp_start = grp_len;
  arma::vec grp_end = arma::cumsum(grp_len) - 1;
  grp_start(0) = 0;
  grp_start.subvec(1, J-1) = arma::cumsum(grp_len.subvec(0,J-2));
  
  arma::vec grad_beta = arma::zeros(p);
  arma::vec beta = beta_0;
  arma::vec eta_minus_k = arma::zeros(p);
  arma::vec beta_old = beta_0;
  
  // grp_start.print("grp_start");
  // grp_end.print("grp_end");
  // beta.subvec(grp_start(0), grp_end(0)).print("beta 0");
  // X.cols(grp_start(0), grp_end(0)).print("X 0");
  int iter = 1;
  do{
    beta_old = beta;
    //beta.print("beta");
    for(int k=0; k < J; k++){
      
      // Check if group k is zero
      eta_minus_k = X*beta - X.cols(grp_start(k), grp_end(k))*beta.subvec(grp_start(k), grp_end(k));
      //eta_minus_k.print("eta_minus_k");
      
      grad_beta = grad_R( beta.subvec(grp_start(k), grp_end(k)),
                          X.cols(grp_start(k), grp_end(k)),
                          eta_minus_k,
                          y);
      //grad_beta.print("grad_beta");
      
      if(arma::norm(soft_thresh(-grad_beta, lambda_1), 2) <= lambda_2*gamma(k)){
        beta.subvec(grp_start(k), grp_end(k)) = arma::zeros(grp_len(k));
        //beta.print("beta");
      }else{
        beta.subvec(grp_start(k), grp_end(k)) = optimization_internal_simon_nesterov(R, 
                    grad_R,
                    Update,
                    beta.subvec(grp_start(k), grp_end(k)),
                    lambda_1,
                    lambda_2,
                    gamma(k),
                    X.cols(grp_start(k), grp_end(k)),
                    eta_minus_k,
                    y,
                    t0);
        //beta.print("beta");
      }
    }
    iter++;
  }while(!arma::approx_equal(beta, beta_old, "absdiff", 0.001) && iter < MAX_ITER);
  return beta;
}


double get_lambda1_max(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X, 
    const arma::mat & y,
    const arma::vec & grp_len
){
  double min_lambda1 = 0;
  int p = X.n_cols;
  int N = X.n_rows;
  int J = grp_len.n_elem;
  
  arma::vec grad;
  arma::vec eta_minus_k = arma::zeros(N);
  
  arma::vec grp_start = grp_len;
  arma::vec grp_end = arma::cumsum(grp_len) - 1;
  grp_start(0) = 0;
  grp_start.subvec(1, J-1) = arma::cumsum(grp_len.subvec(0,J-2));
  
  
  for (int k = 0; k < J; k++)
  {
    arma::vec beta0 = arma::zeros(grp_len(k));
    grad = grad_R(beta0, X.cols(grp_start(k), grp_end(k)), eta_minus_k, y );
    double m = arma::abs(grad).max();
    if (m > min_lambda1){
      min_lambda1 = m;
    }
  }
  
  return min_lambda1;
}
double get_lambda1_max(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X, 
    const arma::mat & y
){
  double min_lambda1 = 0;
  int p = X.n_cols;
  int N = X.n_rows;
  arma::vec grad;
  arma::vec eta_minus_k = arma::zeros(N);
  arma::vec beta0 = arma::zeros(p);
  grad = grad_R(beta0, X, eta_minus_k, y );
  min_lambda1 = arma::abs(grad).max();
  
  return min_lambda1;
}

arma::vec get_lambda2gammak_max(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X, 
    const arma::mat & y,
    const arma::vec & grp_len,
    double lambda1
){
  
  int p = X.n_cols;
  int N = X.n_rows;
  int J = grp_len.n_elem;
  arma::vec lambda2gammak = arma::zeros(J);
  
  arma::vec grad;
  arma::vec eta_minus_k = arma::zeros(N);
  
  arma::vec grp_start = grp_len;
  arma::vec grp_end = arma::cumsum(grp_len) - 1;
  grp_start(0) = 0;
  grp_start.subvec(1, J-1) = arma::cumsum(grp_len.subvec(0,J-2));
  
  for (int k = 0; k < J; k++)
  {
    arma::vec beta0 = arma::zeros(grp_len(k));
    grad = grad_R(beta0, X.cols(grp_start(k), grp_end(k)), eta_minus_k, y );
    lambda2gammak(k) = arma::norm( soft_thresh(grad, lambda1), 2);
  }
  
  return lambda2gammak;
}


Rcpp::List isgl_rs(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    double(*R_v)(const arma::vec &, const arma::mat &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X_t, 
    const arma::mat & y_t,
    const arma::mat & X_v, 
    const arma::mat & y_v,
    const arma::vec & grp_len,
    const int num_iter,
    double t0){
  
  int p = X_t.n_cols;
  int J = grp_len.n_elem;
  
  arma::vec gammak = arma::zeros(J);
  double lambda1 = 0;
  double lambda2 = 1;
  
  arma::vec beta = arma::zeros(p);
  
  arma::vec grp_start = grp_len;
  arma::vec grp_end = arma::cumsum(grp_len) - 1;
  grp_start(0) = 0;
  grp_start.subvec(1, J-1) = arma::cumsum(grp_len.subvec(0,J-2));
  
  double risk_v = 1e32;
  
  Rcpp::List fit = Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("lambda1") = lambda1,
    Rcpp::Named("lambda2") = lambda2,
    Rcpp::Named("gammak") = gammak,
    Rcpp::Named("R_v") = risk_v
  );
  
  arma::vec lambda1_vec = get_lambda1_max(R, grad_R, X_t, y_t, grp_len) * arma::sort(arma::randu(num_iter), "descend");
  
  for (int i = 0; i < num_iter; i++)
  {
    lambda1 = lambda1_vec(i);
    gammak = get_lambda2gammak_max(R, grad_R, X_t, y_t, grp_len, lambda1) % arma::randu(J);
    // Compute the sgl solution
    beta = optimization_external_simon_nesterov(R, grad_R, update_sparse_group_lasso, fit["beta"], lambda1, lambda2, gammak, X_t, y_t, grp_len, t0);
    //beta = optimization_external_simon_nesterov(R, grad_R, update_sparse_group_lasso, beta, lambda1, lambda2, gammak, X_t, y_t, grp_len, t0);
    
    // Compute the validation error
    risk_v = R_v(beta, X_v, y_v);
    
    // Check if this is the best solution
    if(risk_v < Rcpp::as<double>(fit["R_v"])){
      // if beta is optimal
      fit["beta"] = beta;
      fit["lambda1"] = lambda1;
      fit["gammak"] = gammak;
      fit["R_v"] = risk_v; 
    }
  } 
  
  return fit;
}

Rcpp::List il_rs(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    double(*R_v)(const arma::vec &, const arma::mat &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X_t, 
    const arma::mat & y_t,
    const arma::mat & X_v, 
    const arma::mat & y_v,
    const int num_iter,
    double t0){
  
  int p = X_t.n_cols;
  
  double lambda1 = 1;
  double lambda2 = 0;
  
  arma::vec beta = arma::zeros(p);
  
  double risk_v = 1e32;
  
  Rcpp::List fit = Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("lambda1") = lambda1,
    Rcpp::Named("lambda2") = lambda2,
    Rcpp::Named("gammak") = 0,
    Rcpp::Named("R_v") = risk_v
  );
  
  arma::vec lambda1_vec = get_lambda1_max(R, grad_R, X_t, y_t) * arma::sort(arma::randu(num_iter), "descend");
  
  for (int i = 0; i < num_iter; i++)
  {
    lambda1 = lambda1_vec(i);
    // Compute the lasso solution
    arma::vec eta_minus_k = arma::zeros(X_t.n_rows);
    beta = optimization_internal_simon_nesterov(R, grad_R, update_lasso, 
                                                fit["beta"], lambda1, 0, 0, X_t, eta_minus_k,
                                                y_t, t0);
    
    // Compute the validation error
    risk_v = R_v(beta, X_v, y_v);
    
    // Check if this is the best solution
    if(risk_v < Rcpp::as<double>(fit["R_v"])){
      // if beta is optimal
      fit["beta"] = beta;
      fit["lambda1"] = lambda1;
      fit["R_v"] = risk_v; 
    }
  } 
  
  return fit;
}

Rcpp::List igl_rs(
    double(*R)(const arma::vec &, const arma::mat &, const arma::vec &, const arma::mat &),
    double(*R_v)(const arma::vec &, const arma::mat &, const arma::mat &),
    arma::vec(*grad_R)(const arma::vec &, const arma::mat &, const arma::vec &,const arma::mat &),
    const arma::mat & X_t, 
    const arma::mat & y_t,
    const arma::mat & X_v, 
    const arma::mat & y_v,
    const arma::vec & grp_len,
    const int num_iter,
    double t0){
  
  int p = X_t.n_cols;
  int J = grp_len.n_elem;
  
  arma::vec gammak = arma::zeros(J);
  double lambda1 = 0;
  double lambda2 = 1;
  
  arma::vec beta = arma::zeros(p);
  
  arma::vec grp_start = grp_len;
  arma::vec grp_end = arma::cumsum(grp_len) - 1;
  grp_start(0) = 0;
  grp_start.subvec(1, J-1) = arma::cumsum(grp_len.subvec(0,J-2));
  
  double risk_v = 1e32;
  
  Rcpp::List fit = Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("lambda1") = lambda1,
    Rcpp::Named("lambda2") = lambda2,
    Rcpp::Named("gammak") = gammak,
    Rcpp::Named("R_v") = risk_v
  );
  
  
  for (int i = 0; i < num_iter; i++)
  {
    lambda1 = 0;
    gammak = get_lambda2gammak_max(R, grad_R, X_t, y_t, grp_len, lambda1) % arma::randu(J);
    // Compute the sgl solution
    beta = optimization_external_simon_nesterov(R, grad_R, update_sparse_group_lasso, fit["beta"], lambda1, lambda2, gammak, X_t, y_t, grp_len, t0);
    //beta = optimization_external_simon_nesterov(R, grad_R, update_sparse_group_lasso, beta, lambda1, lambda2, gammak, X_t, y_t, grp_len, t0);
    // Compute the validation error
    risk_v = R_v(beta, X_v, y_v);
    
    // Check if this is the best solution
    if(risk_v < Rcpp::as<double>(fit["R_v"])){
      // if beta is optimal
      fit["beta"] = beta;
      fit["gammak"] = gammak;
      fit["R_v"] = risk_v; 
    }
  } 
  
  return fit;
}