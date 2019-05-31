double R_coxph(const arma::vec & beta_k, 
               const arma::mat & X_k,
               const arma::vec & eta_minus_k,
               const arma::mat & y){
  // IMPORTANT! y: is N x 2 with y(,0) = t and y(,1) = sigma
  // In addition, t is in decreasing order.
  
  arma::vec eta = eta_minus_k + X_k*beta_k;
  int N = X_k.n_rows;
  
  // Compute S1
  arma::vec S1 = arma::cumsum(arma::exp(eta));
  
  for(int i = N-1; i>0; i--){
    if(y(i,0) == y(i-1, 0)){
      S1(i-1) = S1(i);
    }
  }
  
  return arma::as_scalar(y.col(1).t()*(-eta + arma::log(S1)));
}
double R_v_coxph(const arma::vec & beta, 
               const arma::mat & X,
               const arma::mat & y){
  // IMPORTANT! y: is N x 2 with y(,0) = t and y(,1) = sigma
  // In addition, t is in decreasing order.
  
  arma::vec eta = X*beta;
  int N = X.n_rows;
  
  // Compute S1
  arma::vec S1 = arma::cumsum(arma::exp(eta));
  
  for(int i = N-1; i>0; i--){
    if(y(i,0) == y(i-1, 0)){
      S1(i-1) = S1(i);
    }
  }
  
  return arma::as_scalar(y.col(1).t()*(-eta + arma::log(S1)));
}

arma::vec grad_R_coxph(const arma::vec & beta_k, 
                       const arma::mat & X_k,
                       const arma::vec & eta_minus_k,
                       const arma::mat & y){
  // IMPORTANT! y: is N x 2 with y(,0) = t and y(,1) = sigma
  // In addition, t is in decreasing order.
  
  arma::vec eta = eta_minus_k + X_k*beta_k;
  int N = X_k.n_rows;
  int p_k = X_k.n_cols;
  
  // Compute S1
  arma::vec S1 = arma::exp(eta);
  
  // Compute S2
  arma::mat S2 = X_k.t();
  for(int i = 0; i < N; i++){
    S2.col(i) *= S1(i);
  }
  S2 = arma::cumsum(S2, 1);
  
  // Update S1
  S1 = arma::cumsum(S1);
  
  // Compute S3
  arma::colvec S3 = arma::zeros<arma::colvec>(p_k);
  
  int i = N-1;
  while(i>=0){
    if(y(i,1)==1){S3 += S2.col(i)/S1(i);}
    // int l = 1;
    // while( i>= l && y(i,0)==y(i-l,0)){
    //   if(y(i-l,1)==1){S3 += S2.col(i)/S1(i);}
    //   l++;
    // }
    // i = i-l;
    i--;
  }
  arma::vec g = (X_k.t()*y.col(1) - S3);
  // double max = arma::abs(g).max();
  //return -g/(max+N);
  return -g;
  // return -X_k.t()*y.col(1) + S3;
}

