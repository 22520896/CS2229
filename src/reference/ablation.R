# Usage: Rscript SCRIPT FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]
suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
data <- read_csv(file.path(cache_dir,"ablations.csv.gz"),show_col_types=FALSE)
f <- figure_number
p <- function(name) data %>% filter(xp==name,!is.na(mean_time)) %>% rename()
set.seed(0)
if (f==18) {
 output <- p("high_frequencies") %>% random_search(variable=transform__2__cov_mult,n_shuffles=15,default_first=T) %>%
   filter(random_rank==60,data__keyword!="poker",model_name!="MLP",transform__2__cov_mult<0.5) %>%
   transmute(data__keyword,model_name,condition=transform__2__cov_mult,score=mean_test_score,shuffle=n_dataset)
}
if (f==20) {
 df_normalized <- p("show_uninformative_features") %>% group_by(data__keyword) %>%
   mutate(mean_val_score=(mean_val_score-0.5)/(max(mean_val_score,na.rm=T)-0.5),
          mean_test_score=(mean_test_score-0.5)/(max(mean_test_score,na.rm=T)-0.5)) %>% ungroup()
 script <- readLines(file.path(upstream_dir,"uninformative_features.R"))
 start <- grep("^N_SHUFFLE <- 30",script)
 finish <- grep("^res_ <- res %>%",script)
 eval(parse(text=script[start:(finish-1)]))
 output <- res_datasets %>% filter(random_rank==20,data__keyword!="poker",model_name=="GradientBoostingTree") %>%
   group_by(data__keyword,method,condition=transform__0__num_features_to_remove) %>%
   summarise(low=min(mean_test_score),high=max(mean_test_score),mean=mean(mean_test_score),
             count=sum(!is.na(mean_test_score)),.groups="drop") %>% filter(count==30)
}
if (f==21 || f==121) {
 shuffle_n <- if(f==21) 15 else 30
 output <- p("add_uninformative_features") %>% filter(!is.na(mean_test_score),!is.na(mean_val_score)) %>%
   random_search(variable=transform__0__multiplier,n_shuffles=shuffle_n,default_first=T) %>%
   filter(!data__keyword %in% c("poker","jannis"),random_rank==20) %>%
   group_by(data__keyword,model_name,condition=transform__0__multiplier-1) %>%
   summarise(low=min(mean_test_score),high=max(mean_test_score),mean=mean(mean_test_score),
             count=sum(!is.na(mean_test_score)),.groups="drop") %>% filter(count==shuffle_n)
}
if (f==22) {
 output <- p("remove_uninformative_features") %>% random_search(variable=transform__0__num_features_to_remove,n_shuffles=30,default_first=T) %>%
   filter(!data__keyword %in% c("poker","jannis"),random_rank==30) %>%
   group_by(data__keyword,model_name,condition=transform__0__num_features_to_remove) %>%
   summarise(low=min(mean_test_score,na.rm=T),high=max(mean_test_score,na.rm=T),mean=mean(mean_test_score,na.rm=T),
             count=sum(!is.na(mean_test_score)),.groups="drop") %>% filter(count>0)
}
if (f==23 || f==24) {
 panels <- lapply(c(0,0.5),function(fraction) {
   p("random_rotation") %>% filter(transform__0__num_features_to_remove==fraction) %>%
     random_search(variable=transform__2__deactivated,n_shuffles=30,default_first=T)
 })
 output <- panels[[f-22]] %>% filter(data__keyword!="poker",random_rank==20,
  model_name %in% c("GradientBoostingTree","RandomForest","FT Transformer","Resnet")) %>%
  transmute(data__keyword,model_name,condition=!transform__2__deactivated,score=mean_test_score,shuffle=n_dataset)
}
write_csv(output,output_file)
