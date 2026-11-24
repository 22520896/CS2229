# Usage: Rscript tuned_estimators.R FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]
suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
stopifnot(figure_number %in% c(39,40))
kind <- if(figure_number==39) "numerical" else "categorical"
read <- function(prefix) {
  read_csv(file.path(cache_dir,paste0(prefix,"_",kind,".csv.gz")),
           show_col_types=FALSE,col_types=cols(.default=col_character())) %>%
    mutate(across(c(mean_time,mean_val_score,mean_test_score),as.numeric))
}
models <- c("GradientBoostingTree","RandomForest","XGBoost")
fixed <- read("time") %>% filter(model_name %in% models,!is.na(mean_time))
tuned <- read("tuned") %>% filter(model_name %in% models,!is.na(mean_time)) %>%
  mutate(model_name=paste0(model_name,"_tune"))
data <- bind_rows(fixed,tuned)
run <- function(task) {
  set.seed(0)
  regression <- task=="regression"
  benchmark_name <- paste0(kind,if(regression) "_regression_medium" else "_classif_medium")
  input <- data %>% filter(benchmark==benchmark_name)
  if(figure_number==39 && regression) input <- filter(input,data__keyword!="isolet")
  n <- n_distinct(input$data__keyword)
  prepared <- input %>% normalize_no_variable(quantile=if(regression) .5 else .1)
  if(regression) prepared <- mutate(prepared,mean_test_score=pmax(mean_test_score,0))
  prepared %>% random_search_no_variable(n_shuffles=15,default_first=T) %>%
    group_by(random_rank,model_name,n_dataset) %>%
    summarise(score=mean(mean_test_score,na.rm=T),.groups="drop_last") %>%
    summarise(mean=mean(score,na.rm=T),low=min(score,na.rm=T),high=max(score,na.rm=T),
              count=sum(!is.na(score)),.groups="drop") %>%
    mutate(low=ifelse(count==0,NA,low),high=ifelse(count==0,NA,high),datasets=n,panel=task) %>%
    dplyr::rename(rank=random_rank)
}
write_csv(bind_rows(lapply(c("classification","regression"),run)),output_file)
