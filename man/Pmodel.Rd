\name{Pmodel}
\alias{Pmodel}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Penalized model set-up
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Configures an ishpR model, preprocessing the input data and setting up the parameters.
}
\usage{
Pmodel(x_train, y_train, x_valid = NULL, y_valid = NULL, validation_split = 0.4, loss = "mse", 
       metric = "mse", regularization = "sparse_group_lasso", scale = F, center = T, 
       scale_type = "sigma", search_method = "bounded_random", grp_idx = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x_train}{
%%     ~~Describe \code{x_train} here~~
  training data (covariates). Should be a \code{matrix} of \code{N} times \code{p}.
}
  \item{y_train}{
%%     ~~Describe \code{y_train} here~~
  training data (response). Should be a \code{matrix} of \code{N} times \code{1} (linear and logistic regressions) or \code{N} times \code{2} (cox regression: time, status 1=failure/0=rigth censored).
}
  \item{x_valid}{
%%     ~~Describe \code{x_valid} here~~
  validation data x (optional)
}
  \item{y_valid}{
%%     ~~Describe \code{y_valid} here~~
  validation data y (optional)
}
  \item{validation_split}{
%%     ~~Describe \code{validation_split} here~~
  if \code{x_valid} and \code{y_valid} are not provided, a proportion of \code{validation_split} from training data is randomly selected for validation.
}
  \item{loss}{
%%     ~~Describe \code{loss} here~~
Loss function to optimize in the regression.
One of the following:

- \code{"mse"} (default) Mean Squared Error. Linear regression.

- \code{"logistic"} Binary cross-entropy. Logistic regression.

- \code{"cox"} Cox proportional hazard models' log-likelihood. Cox regression

}
  \item{metric}{
%%     ~~Describe \code{metric} here~~
Metric function to optimize with respect to the hyper-parameters in the validation data. All losses are available, but non-differentiable functions are also included, such as:

- \code{"acc"} Accuracy. 

- \code{"auc"} Area under the ROC curve. 

- \code{"f1"} The F1 score. 

- \code{"mae"} Mean Absolute Error. 
}
  \item{regularization}{
%%     ~~Describe \code{regularization} here~~
Penalty. Possible choices are:

- \code{"sparse_group_lasso"} (default)

- \code{"lasso"}

- \code{"group_lasso"}
}
  \item{scale}{
%%     ~~Describe \code{scale} here~~
Scale the data before fitting the model?
}
  \item{center}{
%%     ~~Describe \code{center} here~~
Center the data before fitting the moedl?
}
  \item{scale_type}{
%%     ~~Describe \code{scale_type} here~~
If \code{scale = TRUE}, what type of scaling to use. Possible values:

- \code{"sigma"} (default) The standard normalization (center and divide by the deviation)

- \code{"0-1"} Force the columns of \code{x_train} between 0 and 1.
}
  \item{search_method}{
%%     ~~Describe \code{search_method} here~~
Iterative search method to find the optimal hyper-parameters. Possible choices are:

- \code{"bounded_random"} (default) Type of uniform random search in which the possible candidates are chosen inside a non-squared region defined by the theoretical maximum values. The resultant fit is never the null model.

- \code{"iterative"} The custom iterative method described in Laria et. al (2019). 
}
  \item{grp_idx}{
%%     ~~Describe \code{grp_idx} here~~
Group indices. Vector of length \code{p}.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
An object of class \code{ishpR}.
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
Juan C. Laria, M. Carmen Aguilera-Morillo & Rosa E. Lillo (2019) An Iterative Sparse-Group Lasso, Journal of Computational and Graphical Statistics, DOI: 10.1080/10618600.2019.1573687
}
\author{
%%  ~~who you are~~
Juan C. Laria
}
\note{
%%  ~~further notes~~
}

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
\dontrun{
# ============
# Linear model
# ============
N = 100
p = 200

X = matrix(rnorm(N*p), ncol=p)
grp_idx = sample(1:25, p, replace = TRUE)

# linear
y = X[,1:5] \%*\% 1:5 + rnorm(N)

pmodel = Pmodel(x_train = X, y_train = y, grp_idx = grp_idx, 
                center = FALSE, scale = FALSE, scale_type = "sigma")
pmodel = fit(pmodel, num_iter = 20)
pmodel

# =========
# logistic
# =========

N = 100
p = 200

X = matrix(rnorm(N*p), ncol=p)
grp_idx = rep(1:10, each=20)

pr = (1 + exp(-X[,1:5]\%*\%1:5))^(-1)
y = as.matrix(rbinom(N, size=1, prob = pr), ncol = 1)

pmodel = Pmodel(x_train = X, y_train = y, grp_idx = grp_idx, center = FALSE, scale = FALSE, 
                scale_type = "sigma", loss = "logistic",
                metric = "logistic", validation_split = 0.4)
pmodel = fit(pmodel, num_iter = 20)
pmodel
table(predict(pmodel, X, type = "class"), y)


# ===============
# Cox regression
# ===============

N=100;p=200
grp_idx = rep(1:10, each=20)
nzc=5
X=matrix(rnorm(N*p),N,p)
beta=1:nzc
fx=X[,seq(nzc)]\%*\%beta
hx=exp(fx)
ty=rexp(N,hx)
cy = rexp(N, hx/2)
tcens = rep(0, N)
tcens[cy < ty] = 1
y=cbind(time=pmin(ty, cy), status = 1 - tcens)

pmodel = Pmodel(x_train = X, y_train = y, grp_idx = grp_idx, center = FALSE, scale = FALSE, 
                scale_type = "sigma", loss = "cox", 
                metric = "cox", validation_split = 0.4)
pmodel = fit(pmodel, num_iter = 9)
pmodel
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line

