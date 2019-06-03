Pmodel <- function(x_train,
                   y_train,
                   x_valid=NULL,
                   y_valid=NULL,
                   validation_split=0.4,
                   loss = "mse", # mse, logistic, cox, poisson
                   metric = "mse", # mse, mae, logistic, acc, cox, poisson 
                   regularization = "sparse_group_lasso", # lasso, group_lasso, sparse_group_lasso
                   scale = F,
                   center = T,
                   scale_type = "sigma", # "0-1" "sigma",
                   search_method = "bounded_random", # "bounded_random", "iterative" ,
                   grp_idx = NULL
){
  pmodel = list(
    x_train=NULL,
    y_train=NULL,
    x_valid=NULL,
    y_valid=NULL,
    validation_split=validation_split,
    loss = loss,
    metric = metric,
    regularization = regularization,
    scale = scale,
    center = center,
    scale_type = scale_type,
    beta = NULL,
    intercept = 0,
    search_method = search_method,
    grp_len = NULL,
    ord_col = NULL,
    lambda1 = NULL,
    lambda2 = NULL,
    gamma = NULL,
    R_v = 1e32,
    cutpoint = 0.5
  )
  if(is.null(x_valid) & !is.null(validation_split)){
    valid_idx = sample(nrow(x_train), floor(nrow(x_train)*validation_split))
    
    x_valid = x_train[valid_idx, ]
    x_train = x_train[-valid_idx,]
    y_valid = y_train[valid_idx, ]
    y_train = y_train[-valid_idx,]
  }
  
  if(center){
    X = rbind(x_train, x_valid)
    X = scale(X, center = T, scale = F)
    
    x_train = X[1:nrow(x_train),]
    if(!is.null(x_valid)) x_valid = X[(nrow(x_train)+1):nrow(X), ]
    
    attr(pmodel, "scaled:center") = attr(X, "scaled:center")
  }
  
  if(scale){
    X = rbind(x_train, x_valid)
    
    if(scale_type=="0-1"){
      min_ = apply(X, 2, min)
      max_ = apply(X, 2, max)
      X = t(apply(X, 1, function(x){
        (x-min_)/(max_ - min_)
      }))
      
      attr(pmodel, "scaled:min") = min_
      attr(pmodel, "scaled:max") = max_
      
      x_train = X[1:nrow(x_train),]
      if(!is.null(x_valid)) x_valid = X[(nrow(x_train)+1):nrow(X), ]
      
    }else if(scale_type=="sigma"){
      X = scale(X, center = F, scale = T)
      attr(pmodel, "scaled:scale") = attr(X, "scaled:scale")
      
      x_train = X[1:nrow(x_train),]
      if(!is.null(x_valid)) x_valid = X[(nrow(x_train)+1):nrow(X), ]
    }
    
  }
  
  # Order x
  ord = order(grp_idx)
  grp_len = tabulate(grp_idx[ord])
  x_train = x_train[ ,ord]
  x_valid = x_valid[, ord]
  
  pmodel$ord_col = ord
  pmodel$grp_len = grp_len
  
  # Order y
  if (loss == "cox") {
    ord = order(y_train[,1], decreasing = T)
    x_train = x_train[ord, ]
    y_train = y_train[ord, ]
    
    ord = order(y_valid[,1], decreasing = T)
    x_valid = x_valid[ord, ]
    y_valid = y_valid[ord, ]
  }
  
  pmodel$x_train = as.matrix(x_train)
  pmodel$y_train = as.matrix(y_train)
  pmodel$x_valid = as.matrix(x_valid)
  pmodel$y_valid = as.matrix(y_valid)
  
  class(pmodel) = "ishpR"
  
  return(pmodel)
}


fit = function(pmodel, num_iter = 100){
  LOSS_MSE = 0
  LOSS_LOGISTIC = 1
  LOSS_COX = 2
  REGULARIZATION_SGL = 0
  REGULARIZATION_LASSO = 1
  REGULARIZATION_GLASSO = 2
  # loss
  switch(pmodel$loss, 
         mse = {
           loss = LOSS_MSE
         },
         logistic = {
           loss = LOSS_LOGISTIC
         },
         cox = {
           loss = LOSS_COX
         })
  
  # metric
  switch(pmodel$metric, 
         mse = {
           metric = LOSS_MSE
         },
         logistic = {
           metric = LOSS_LOGISTIC
         },
         cox = {
           metric = LOSS_COX
         })
  
  # regularization
  switch (pmodel$regularization,
    sparse_group_lasso = {
      regularization = REGULARIZATION_SGL
    },
    lasso = {
      regularization = REGULARIZATION_LASSO
    },
    group_lasso = {
      regularization = REGULARIZATION_GLASSO
    }
  )
  
  # Compute intercept linear case
  if (pmodel$loss == "mse") {
    pmodel$intercept = mean(pmodel$y_train)
    pmodel$y_train = pmodel$y_train - pmodel$intercept
    pmodel$y_valid = pmodel$y_valid - pmodel$intercept
  }
  
  obj = isglasso(pmodel$x_train,
                pmodel$y_train,
                pmodel$x_valid,
                pmodel$y_valid,
                pmodel$grp_len,
                num_iter,
                loss,
                metric,
                regularization)
  
  pmodel$beta = obj$beta
  pmodel$lambda1 = obj$lambda1
  pmodel$lambda2 = obj$lambda2
  pmodel$gamma = obj$gammak
  pmodel$R_v = obj$R_v
  
  # Reset data linear case
  if (pmodel$loss == "mse") {
    pmodel$y_train = pmodel$y_train + pmodel$intercept
    pmodel$y_valid = pmodel$y_valid + pmodel$intercept
  }
  
  # Find intercept logistic case
  if (pmodel$loss == "logistic") {
      pmodel$intercept = log(mean(pmodel$y_train)/(1 - mean(pmodel$y_train)))
  }
  
  return(pmodel)
}

predict.ishpR = function(pmodel, 
                         x_test, 
                         type = "default", # probs, response, class
                         transform = TRUE
                         ){
  if (transform) {
    if(pmodel$center){
      x_test = x_test - rep(1, nrow(x_test)) %o% attr(pmodel, "scaled:center")
    }
    if(pmodel$scale){
      if(pmodel$scale_type == "sigma"){
        x_test = x_test %*% diag(1/attr(pmodel, "scaled:scale"))
      }
      if(pmodel$scale_type == "0-1"){
        x_test = t(apply(x_test, 1, function(x){(x - attr(pmodel, "scaled:min"))/(attr(pmodel, "scaled:max") - attr(pmodel, "scaled:min"))}))
      }
    }
  }
  if (type == "default") {
    # Try to figure it out
    switch (pmodel$metric,
      mse = {type = "response"},
      logistic = {type = "probs"},
      acc = {type = "class"},
      {
        type = "response"
      }
    )
  }
  
  eta = pmodel$intercept + x_test %*% pmodel$beta
  if (type == "response") {
    return(eta)
  }
  if (type == "probs") {
    p = (1 + exp(-eta))^-1
    return(p)
  }
  if (type == "class") {
    p = (1 + exp(-eta))^-1
    y_pred = (p > pmodel$cutpoint) + 0
    return(y_pred)
  }
}

print.ishpR = function(pmodel){
  cat("ishpR object\n")
  cat(paste0("x_train: ", nrow(pmodel$x_train), " obs. x ", ncol(pmodel$x_train), " vars. \n"))
  cat(paste0("x_valid: ", nrow(pmodel$x_valid), " obs. x ", ncol(pmodel$x_valid), " vars. \n"))
  cat(paste0("Loss: ", pmodel$loss, "\n"))
  cat(paste0("Metric: ", pmodel$metric, "\n"))
  cat("Solution: \n")
  if (pmodel$R_v >= 1e32) {
    cat("Model was not fitted.")
  }else{
    cat(paste0(" Method: ", pmodel$search_method, " search\n"))
    cat(paste0(" Regularization: ", pmodel$regularization), "\n")
    cat(" beta: \n")
    cat("\t var \t coef. \n")
    for (v in which(abs(pmodel$beta) > 0)) {
      cat(paste0("\t ", v," \t ", round(pmodel$beta[v], 4), " \n"))
    }
    
    cat(paste0(" intercept: ", round(pmodel$intercept, 4), "\n"))
    cat(paste0(" metric (validation): ", round(pmodel$R_v, 4), "\n"))
    cat(" hyper-parameters: \n")
    cat(paste0("  lambda_1: ", round(pmodel$lambda1, 4), "\n"))
    cat(paste0("  lambda_2: ", round(pmodel$lambda2, 4), "\n"))
    if(pmodel$regularization != "lasso"){
      cat("  gamma:\n")
      for (v in unique(rep(1:length(pmodel$grp_len), times = pmodel$grp_len)[abs(pmodel$beta) > 0])) {
        cat(paste0("\t ", v," \t ", round(pmodel$gamma[v], 4), " \n"))
      } 
    }
  }
}

plot.ishpR = function(pmodel, 
                      type = "groups"
                      ){
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("Package \"ggplot2\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  if (!requireNamespace("gridExtra", quietly = TRUE)) {
    stop("Package \"gridExtra\" needed for this function to work. Please install it.",
         call. = FALSE)
  }
  switch (type,
    groups = {
        included_ = sum(abs(pmodel$beta) > 0)
        excluded_ = length(pmodel$beta) - included_
        
        plot_all = ggplot2::ggplot()+
          ggplot2::aes(x = "", y = c(included_, excluded_),
              fill = c("included", "excluded"))+
          ggplot2::geom_bar(stat = "identity")+
          ggplot2::coord_flip()+
          ggplot2::labs(y = "Number of variables", 
               title = "Overall view")+
          ggplot2::guides(fill = F)+
          ggplot2::scale_fill_brewer(palette = "Greys")+
          ggplot2::theme_minimal()
        
      if(pmodel$regularization == "lasso"){
          plot_all
      }else{
        df = NULL
        grp_start = c(1, cumsum(pmodel$grp_len) + 1)
        grp_end = cumsum(pmodel$grp_len)
        for (k in 1:length(pmodel$grp_len)) {
          included = sum(abs(pmodel$beta[grp_start[k]:grp_end[k]])>0)
          excluded = pmodel$grp_len[k] - included
          if(included == 0) next()
          df = rbind(df,
                     data.frame(
                       gr = rep(paste0("Group ", k), 2),
                       count = c(included, excluded),
                       type = c("included", "excluded")
                     ))
        }
        l = length(unique(df$gr))
        if(l <= 4){ 
          col = 1
        }else if(l <= 8){
          col = 2
        }else{
          col = 3
        }
        
        plot_gr  = ggplot2::ggplot(df)+
          ggplot2::aes(x = "", y = count, 
              fill = type)+
          ggplot2::geom_bar(stat = "identity")+
          ggplot2::coord_flip()+
          ggplot2::labs(y = "", fill = "Status", title = "Groupwise view")+
          ggplot2::theme_minimal()+
          ggplot2::theme(legend.position = "top")+
          ggplot2::scale_fill_brewer(palette = "Greys")+
          ggplot2::facet_wrap(~gr, ncol = col)
        gridExtra::grid.arrange(plot_gr, plot_all, heights = c(3,1))
      }
    }
  )
}

split_data = function(pmodel){
  X = rbind(pmodel$x_train, pmodel$x_valid)
  y = rbind(pmodel$y_train, pmodel$y_valid)
  
  valid_idx = sample(nrow(X), floor(nrow(X)*pmodel$validation_split))
  pmodel$x_train = as.matrix(X[-valid_idx, ])
  pmodel$y_train = as.matrix(y[-valid_idx, ])
  pmodel$x_valid = as.matrix(X[valid_idx, ])
  pmodel$y_valid = as.matrix(y[valid_idx, ])
  
  # Order y
  if (pmodel$loss == "cox") {
    ord = order(pmodel$y_train[,1], decreasing = T)
    pmodel$x_train = pmodel$x_train[ord, ]
    pmodel$y_train = pmodel$y_train[ord, ]
    
    ord = order(pmodel$y_valid[,1], decreasing = T)
    pmodel$x_valid = pmodel$x_valid[ord, ]
    pmodel$y_valid = pmodel$y_valid[ord, ]
  }
  
  return(pmodel)
}

regularization = function(pmodel, regularization = NULL){
  if(is.null(regularization)){
    return(pmodel$regularization)
  }else{
    pmodel$regularization = regularization
    return(pmodel)
  }
}
