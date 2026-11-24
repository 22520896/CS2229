# Usage: Rscript SCRIPT FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]
suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
get <- function(name) read_csv(file.path(cache_dir,paste0(name,".csv.gz")),show_col_types=FALSE, col_types=cols(.default=col_character())) %>% mutate(across(any_of(c("mean_val_score","mean_test_score","mean_time","mean_r2_val","mean_r2_test","step","data__regression","data__categorical")), as.numeric))
reg <- figure_number==34
baseline <- get("time_numerical") %>%
 filter(benchmark==if(reg) "numerical_regression_medium" else "numerical_classif_medium",
        model_name %in% c("Resnet","FT Transformer","GradientBoostingTree","XGBoost"))
b2 <- get("bo_trees_2")
bo <- bind_rows(get("bo_nn"),get("bo_trees")) %>% filter(!data__keyword %in% b2$data__keyword) %>%
 bind_rows(get("bo_nn_2"),b2) %>% rename() %>% filter(data__regression==as.numeric(reg))
if(reg) bo <- mutate(bo, mean_val_score=mean_r2_val,mean_test_score=mean_r2_test)
bo <- bind_rows(bo, baseline %>% filter(hp=="default") %>% mutate(step=0))
input <- bind_rows(mutate(baseline,bo=F),mutate(bo,bo=T)) %>%
 filter(!is.na(mean_time),!data__keyword %in% (if(reg) c("isolet") else c("california","house_16H"))) %>%
 normalize_no_variable(quantile=if(reg) .5 else .1)
if(reg) input <- mutate(input,mean_test_score=if_else(mean_test_score<0,0,mean_test_score))
set.seed(0)
random <- input %>% filter(!bo) %>% random_search_no_variable(n_shuffles=15,default_first=T) %>%
 filter(random_rank<=200) %>% group_by(random_rank,model_name,n_dataset) %>%
 summarise(score=mean(mean_test_score,na.rm=T),.groups="drop_last") %>%
 summarise(mean=mean(score,na.rm=T),low=min(score,na.rm=T),high=max(score,na.rm=T),
  count=sum(!is.na(score)),.groups="drop") %>%
 mutate(low=ifelse(count==0,NA,low),high=ifelse(count==0,NA,high),method="Random")
bayesian <- input %>% filter(bo) %>% mutate(random_rank=step) %>%
 group_by(model_name,data__keyword) %>% arrange(random_rank) %>%
 mutate(best=cummax(mean_val_score),mean_test_score=if_else(mean_val_score==best,mean_test_score,NA_real_)) %>%
 group_by(model_name,data__keyword,mean_val_score) %>%
 mutate(mean_test_score=if_else(random_rank==min(random_rank),mean_test_score,NA_real_)) %>% ungroup() %>%
 complete(model_name,data__keyword,random_rank) %>% group_by(model_name,data__keyword) %>%
 mutate(mean_test_score=na.locf(mean_test_score,na.rm=F)) %>% ungroup() %>%
 group_by(model_name,random_rank) %>% summarise(mean=mean(mean_test_score,na.rm=T),.groups="drop") %>%
 filter(random_rank<=200) %>% mutate(method="Bayesian")
write_csv(bind_rows(random,bayesian) %>% dplyr::rename(rank=random_rank),output_file)
