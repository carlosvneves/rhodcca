#' rhodcca
#'
#' Calculate rhodcca
#'
#' @param series1 numeric vector
#' @param series2 numeric vector
#' @param scale_min an integer indicating the minimum scale to be resovled
#' @param scale_max an integer indicating the maximum scale to be resovled
#' @param order is an integer indicating the polynomial order used for
#' detrending the local windows (e.g, 1 = linear, 2 = quadratic, etc.).
#' @param scale_ratio a double indicating the ratio by which scale successive
#' scales. For example, scale_ratio = 2 would create a scales increasing by
#' a power of 2.
#'
#' @return A list of dcca.out (dataframe with columns scales, rho) and error
#'
#' @examples
#' rhodcca.out <- rhodcca(series1, series2)
#'
#' @export
rhodcca <- function(
  series1,
  series2,
  scale_min = NULL,
  scale_max = NULL,
  order = 1,
  scale_ratio = 1.05
) {
  if (length(series1) != length(series2)) {
    stop
  }
  L <- length(series1)
  if (is.null(scale_min)) {
    scale_min <- 3
  }
  if (is.null(scale_max)) {
    scale_max <- as.integer(ceiling(L / 3))
  }

  scales <- rhodcca::logscale(
    scale_min = scale_min,
    scale_max = scale_max,
    scale_ratio = scale_ratio
  )
  dcca.out <- rhodcca::dcca(
    x = series1,
    y = series2,
    order = order,
    scales = scales
  )

  list(
    dcca.out = as.data.frame(list(
      scales = dcca.out$scales,
      rho = dcca.out$rho
    )),
    error = sd(dcca.out$rho) / sqrt(length(dcca.out$rho))
  )
}
