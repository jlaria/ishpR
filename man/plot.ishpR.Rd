\name{plot.ishpR}
\alias{plot.ishpR}
\alias{plot}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Plot function
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
Plots an \code{ishpR} object, using \code{ggplot2}.
}
\usage{
plot.ishpR(pmodel, type = "groups")
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pmodel}{
%%     ~~Describe \code{pmodel} here~~
The \code{ishpR} model.
}
  \item{type}{
%%     ~~Describe \code{type} here~~
Plot type:

- \code{"groups"} Plots the non-zero groups in the model.
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
\code{NULL}
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
}
\examples{
\dontrun{
library(ishpR)
library(dplyr)
library(ggplot2)
library(gridExtra)

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
plot(pmodel)
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
