# Usage: Rscript SCRIPT FIGURE UPSTREAM_ANALYSES PROJECTION_DIR OUTPUT.csv
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4L)
figure_number <- as.integer(args[1])
upstream_dir <- args[2]
cache_dir <- args[3]
output_file <- args[4]
suppressPackageStartupMessages(source(file.path(upstream_dir,"random_search_utils.R")))
x <- read_csv(file.path(cache_dir,"ablations.csv.gz"), show_col_types=FALSE)
set.seed(0)
if (figure_number == 2){
  d <- x %>% filter(xp=="high_frequencies", !is.na(mean_time)) %>%
    rename() %>% normalize(variable=transform__2__cov_mult, normalization_type="quantile", quantile=0.1) %>%
    random_search(variable=transform__2__cov_mult, n_shuffles=15, default_first=T) %>%
    filter(random_rank==60, model_name!="MLP", data__keyword!="poker") %>%
    mutate(dataset=data__keyword, lengthscale=transform__2__cov_mult)
  fit <- lm(mean_test_score ~ dataset + model_name + lengthscale + model_name * lengthscale, data=d)
}
if (figure_number == 3){
  d <- x %>% filter(xp=="remove_uninformative_features", !is.na(mean_time)) %>%
    rename() %>% normalize(variable=transform__0__num_features_to_remove, normalization_type="quantile", quantile=0.1) %>%
    random_search(variable=transform__0__num_features_to_remove, n_shuffles=30, default_first=T) %>%
    filter(random_rank==30, !data__keyword %in% c("poker","jannis")) %>%
    mutate(dataset=data__keyword, prop_removed=transform__0__num_features_to_remove)
  fit <- lm(mean_test_score ~ dataset + model_name + prop_removed + model_name * prop_removed, data=d)
}
if (figure_number == 4){
  d <- x %>% filter(xp=="add_uninformative_features", !is.na(mean_time), !is.na(mean_test_score), !is.na(mean_val_score), !is.na(model_name)) %>%
    rename() %>% normalize(variable=transform__0__multiplier, normalization_type="quantile", quantile=0.1) %>%
    random_search(variable=transform__0__multiplier, n_shuffles=15, default_first=T) %>%
    filter(random_rank==20, !data__keyword %in% c("poker","jannis")) %>%
    mutate(dataset=data__keyword, prop_added=transform__0__multiplier)
  fit <- lm(mean_test_score ~ dataset + model_name + prop_added + model_name * prop_added, data=d)
}
if (figure_number == 5){
  d <- bind_rows(lapply(c(0,0.5), function(fraction) {
    x %>% filter(xp=="random_rotation", transform__0__num_features_to_remove==fraction, !is.na(mean_time)) %>%
      rename() %>% normalize(variable=transform__2__deactivated, normalization_type="quantile", quantile=0.1) %>%
      random_search(variable=transform__2__deactivated, n_shuffles=30, default_first=T) %>%
      mutate(transform__0__num_features_to_remove=fraction)
  })) %>% filter(random_rank==20,data__keyword!="poker") %>%
    mutate(dataset=data__keyword, rotation=!transform__2__deactivated, features_removed=transform__0__num_features_to_remove!=0)
  fit <- lm(mean_test_score ~ dataset + model_name + rotation + features_removed + rotation * model_name + rotation * features_removed, data=d)
}
matrix <- summary(fit)$coefficients
table <- as_tibble(matrix, rownames="term")
colnames(table) <- c("term","estimate","std_error","statistic","p_value")
output <- table %>% filter(!startsWith(term,"dataset")) %>%
  mutate(nobs=nobs(fit), residual_df=df.residual(fit), dataset_baseline=fit$xlevels$dataset[1], model_baseline=fit$xlevels$model_name[1])
write_csv(output,output_file)
