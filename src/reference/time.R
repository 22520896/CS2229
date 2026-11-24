# Usage: Rscript SCRIPT FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]

suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
analysis_kind <- if (figure_number == 6) "numerical" else "categorical"
data <- read_csv(file.path(cache_dir,paste0("time_",analysis_kind,".csv.gz")), show_col_types=FALSE)
keys <- paste0(analysis_kind,c("_classif_medium","_regression_medium"))
frames <- lapply(keys, function(key) {
  set.seed(0)
  panel <- data %>% filter(benchmark==key,!is.na(mean_time))
  n_tasks <- n_distinct(panel$data__keyword)
  is_regression <- grepl("_regression_",key)
  q <- if (!is_regression) 0.1 else if (figure_number==6) 0.5 else 0.4
  cleaned <- panel %>% normalize_no_variable(normalization_type="quantile",quantile=q)
  if (is_regression) cleaned <- cleaned %>% mutate(mean_test_score=if_else(mean_test_score<0,0,mean_test_score))
  cleaned %>%
    random_search_no_variable_time(n_shuffles=15,default_first=T) %>%
    group_by(model_name,cum_time_factor,n_dataset) %>%
    summarise(score=mean(mean_test_score),seconds=median(cum_time,na.rm=T),.groups="drop_last") %>%
    summarise(high=max(score),low=min(score),mean=mean(score),seconds=mean(seconds),.groups="drop") %>%
    filter(!is.na(mean)) %>%
    dplyr::rename(time_bin=cum_time_factor) %>%
    mutate(panel=key,datasets=n_tasks)
})
write_csv(bind_rows(frames),output_file)
