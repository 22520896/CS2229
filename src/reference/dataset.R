# Usage: Rscript SCRIPT FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]
suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
f <- figure_number
modern <- c("numerical_classification_medium","numerical_regression_medium",
            "categorical_classification_medium","categorical_regression_medium")
historical <- sub("classification","classif",modern)
kinds <- c("numerical","numerical","categorical","categorical")
read <- function(name) read_csv(file.path(cache_dir,paste0(name,".csv.gz")), show_col_types=FALSE, col_types=cols(.default=col_character())) %>% mutate(across(any_of(c("mean_val_score","mean_test_score","mean_time","mean_r2_val","mean_r2_test","step","data__regression","data__categorical")), as.numeric))
large <- function(i) read(paste0("large_",kinds[i])) %>% filter(benchmark==sub("medium","large",historical[i]))
clip <- function(dt, regression) {
  if(regression) mutate(dt, mean_test_score=if_else(mean_test_score<0,0,mean_test_score)) else dt
}
raw <- function(input, is_regression) {
  set.seed(0)
  maxima <- input %>% count(data__keyword,model_name) %>% group_by(data__keyword) %>% summarise(min_count=min(n))
  cleaned <- input %>% filter(!is.na(mean_time)) %>% rename() %>%
    mutate(across(where(is.numeric), ~ifelse(is.infinite(.x), NA, .x)))
  cleaned <- clip(cleaned, is_regression)
  cleaned %>% random_search_no_variable(n_shuffles=15,default_first=T,equalize_n_iteration=T) %>%
    left_join(maxima,by="data__keyword") %>% filter(random_rank<min_count,random_rank<=400) %>%
    group_by(data__keyword,model_name,random_rank) %>%
    summarise(mean=mean(mean_test_score,na.rm=T),low=min(mean_test_score,na.rm=T),
              high=max(mean_test_score,na.rm=T),count=sum(!is.na(mean_test_score)),.groups="drop") %>%
    filter(count>0) %>% group_by(data__keyword) %>% mutate(low=pmax(low,min(mean))) %>%
    ungroup() %>% dplyr::rename(rank=random_rank)
}
curve <- function(input,q, is_regression) {
  set.seed(0)
  cleaned <- input %>% filter(!is.na(mean_time)) %>% normalize_no_variable(quantile=q)
  cleaned <- clip(cleaned,is_regression)
  cleaned %>% random_search_no_variable(n_shuffles=15,default_first=T,equalize_n_iteration=T) %>%
    filter(random_rank<=400) %>% group_by(random_rank,model_name,n_dataset) %>%
    summarise(score=mean(mean_test_score,na.rm=T),.groups="drop_last") %>%
    summarise(mean=mean(score,na.rm=T),low=min(score,na.rm=T),high=max(score,na.rm=T),
              count=sum(!is.na(score)),.groups="drop") %>%
    mutate(low=ifelse(count==0,NA,low),high=ifelse(count==0,NA,high)) %>% dplyr::rename(rank=random_rank)
}
if(f>=8 && f<=11){
  i <- f-7; l <- large(i)
  m <- read(paste0("time_",kinds[i])) %>% filter(benchmark==historical[i],data__keyword %in% l$data__keyword)
  if(i!=1) m <- filter(m,model_name %in% l$model_name)
  n <- n_distinct(l$data__keyword)
  output <- bind_rows(curve(m,c(.1,.4,.1,.5)[i],i%%2==0) %>% mutate(train_cap=10000L,datasets=n),
                      curve(l,c(.1,.4,.2,.5)[i],i%%2==0) %>% mutate(train_cap=50000L,datasets=n))
} else if(f>=12 && f<=15){
  i <- f-11
  d <- read("benchmark") %>% filter(benchmark==modern[i])
  if(f==13) d <- filter(d,model_name!="HistGradientBoostingTree")
  output <- raw(d,i%%2==0)
} else {
  i <- if(f==16) 1 else 3
  output <- bind_rows(raw(large(i),FALSE) %>% mutate(panel=sub("medium","large",historical[i])),
                      raw(large(i+1),TRUE) %>% mutate(panel=sub("medium","large",historical[i+1])))
}
write_csv(output,output_file)
