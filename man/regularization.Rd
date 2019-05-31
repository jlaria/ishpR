\name{regularization}
\alias{regularization}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Set or gets the regularization method.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Set or gets the regularization of an \code{ishpR} object. Helpful when comparing different penalizations with the same data.
}
\usage{
regularization(pmodel, regularization = NULL)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pmodel}{
%%     ~~Describe \code{pmodel} here~~
\code{ishpR} object
}
  \item{regularization}{
%%     ~~Describe \code{regularization} here~~
 One of \code{"sparse_group_lasso", "group_lasso", "lasso"}.
}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
If \code{regularization} is missing, this function returns the regularization. Otherwise, it returns an \code{ishpR} with the desired regularization.
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
%%  ~~who you are~~
Juan C. Laria
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
\code{\link{Pmodel}}, \code{\link{fit}}
}
\examples{
\dontrun{
library(ishpR)
library(dplyr)

# linear
N = 100
p = 200
X = matrix(rnorm(N*p), ncol=p)
grp_idx = rep(1:10, each=20)
y = X[,1:5] \%*\% 1:5 + rnorm(N)

pmodel = Pmodel(x_train = X, y_train = y, grp_idx = grp_idx, center = F, scale = F, scale_type = "0-1",
                regularization = "sparse_group_lasso",
                loss = "mse",
                metric = "mse") \%>\% 
         fit(num_iter = 20)
pmodel
pmodel \%>\% regularization("lasso") \%>\% fit(num_iter = 20)
pmodel \%>\% regularization("group_lasso") \%>\% fit(num_iter = 20)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
