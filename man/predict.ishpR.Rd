\name{predict.ishpR}
\alias{predict.ishpR}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
%%  ~~function to do ... ~~
Predicts with an \code{ishpR} model.
}
\description{
%%  ~~ A concise (1-5 lines) description of what the function does. ~~
\code{predict} computes predictions from an object of type \code{ishpR}.
}
\usage{
predict.ishpR(pmodel, x_test, type = "default", transform = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{pmodel}{
%%     ~~Describe \code{pmodel} here~~
The \code{ishpR} model.
}
  \item{x_test}{
%%     ~~Describe \code{x_test} here~~
New \code{x} data.
}
  \item{type}{
%%     ~~Describe \code{type} here~~
The type of prediction. If empty, it is deduced from the loss function. Possible values are:

- \code{"default"}

- \code{"probs"}

- \code{"response"}

- \code{"class"}
}
  \item{transform}{
%%     ~~Describe \code{transform} here~~
Should the matrix \code{x_test} be transformed to match \code{pmodel}'s internal data? (\code{default = TRUE})
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
Vector of predictions.
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
\code{\link{Pmodel}}, \code{\link{fit} , \code{\link{plot.ishpR}}}
}
\examples{
\dontrun{
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
predict(pmodel, X, type = "class")
predict(pmodel, X, type = "probs")
predict(pmodel, X, type = "response")
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ ~kwd1 }% use one of  RShowDoc("KEYWORDS")
\keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
