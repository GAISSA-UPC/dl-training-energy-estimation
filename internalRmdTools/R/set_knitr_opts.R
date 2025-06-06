#' Custom Knit function for RStudio
#'
#' @export
knit_to_reports <- function(input, ...) {
  print(paste("Rendering", input, "to reports directory..."))

  print(getwd())
  rmarkdown::render(
    input,
    output_dir = "../reports",
    envir = globalenv()
  )
}
