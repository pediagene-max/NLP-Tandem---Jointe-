packages <- c("clustrd")

installed <- rownames(installed.packages())
missing <- setdiff(packages, installed)

if (length(missing) > 0) {
  install.packages(missing, repos = "https://cloud.r-project.org")
}

message("R dependencies are ready.")
