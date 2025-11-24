# Walmart Competition #4322
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(themis)
library(DataExplorer)

# Parallel
num_cores <- parallel::detectCores()
cl <- makePSOCKcluster(num_cores - 1)
registerDoParallel(cl)

# Load
trainData <- vroom("train.csv")
testData  <- vroom("test.csv")
features <- vroom("features.csv")
stores <- vroom("stores.csv")

#########
## EDA ##
#########
plot_missing(features)
plot_missing(testData)

# Correct joins
trainData <- trainData %>%
  left_join(features, by = c("Store","Date")) %>%
  left_join(stores, by = "Store")

testData <- testData %>%
  left_join(features, by = c("Store","Date")) %>%
  left_join(stores, by = "Store")

# Markdown fixes
trainData <- trainData %>%
  mutate(across(starts_with("MarkDown"), ~replace_na(., 0))) %>%
  mutate(TotalMarkdown = MarkDown1 + MarkDown2 + MarkDown3 + 
           MarkDown4 + MarkDown5) %>%
  mutate(MarkdownFlag = if_else(TotalMarkdown > 0, 1, 0)) %>%
  select(-IsHoliday.y) %>%
  mutate(IsHoliday.x = factor(IsHoliday.x))

testData <- testData %>%
  mutate(across(starts_with("MarkDown"), ~replace_na(., 0))) %>%
  mutate(TotalMarkdown = MarkDown1 + MarkDown2 + MarkDown3 + 
           MarkDown4 + MarkDown5) %>%
  mutate(MarkdownFlag = if_else(TotalMarkdown > 0, 1, 0)) %>%
  select(-IsHoliday.y) %>%
  mutate(IsHoliday.x = factor(IsHoliday.x))

####################################################################
##################### RECIPE SECTION ###############################
####################################################################

my_recipe <- recipe(Weekly_Sales ~ ., data = trainData) %>%
  step_zv(all_predictors()) %>%
  step_mutate(DecDate = decimal_date(Date)) %>%
  update_role(Store, Dept, Date, new_role = "ID") %>%
  step_mutate(Store = factor(Store),
              Dept = factor(Dept)) %>%
  step_date(Date, features = c("year","week","dow")) %>%
  step_impute_bag(any_of(c("CPI","Unemployment")), impute_with = imp_vars(DecDate, Store)) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

####################################################################
##################### GLOBAL CV ERROR BLOCK ########################
################### (For LearningSuite Report) #####################
####################################################################

set.seed(123)

folds_full <- vfold_cv(trainData, v = 3)

rf_model <- rand_forest(
  trees = tune(),
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger") %>%
  set_mode("regression")

wf_full <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)

grid_full <- grid_regular(
  mtry(range = c(5, 20)),
  trees(range = c(10, 40)),
  min_n(range = c(2, 10)),
  levels = 4
)

# ---- RUN TUNING (this gives the number you report) ----
tuned_results_full <- tune_grid(
  wf_full,
  resamples = folds_full,
  grid = grid_full,
  metrics = metric_set(rmse)
)

# Best parameters
best_params_full <- select_best(tuned_results_full, "rmse")
print(best_params_full)

# Cross-validated RMSE to report on LearningSuite
cv_rmse_to_report <- show_best(tuned_results_full, metric = "rmse", n = 1)
print(cv_rmse_to_report)

####################################################################
##################### STORE/DEPT LOOP FOR KAGGLE ###################
####################################################################

glmnet_model <- linear_reg(penalty = 0.1, mixture = 0.5) %>% 
  set_engine("glmnet")

all_predictions <- list()

groups <- testData %>% distinct(Store, Dept)

for (i in 1:nrow(groups)) {
  store_i <- groups$Store[i]
  dept_i  <- groups$Dept[i]
  
  train_subset <- trainData %>% filter(Store == store_i, Dept == dept_i)
  test_subset  <- testData  %>% filter(Store == store_i, Dept == dept_i)
  
  which_na <- sapply(train_subset, function(x) sum(is.na(x)))
  print(paste("Store:", store_i, "Dept:", dept_i))
  print(which_na[which_na > 0])
  
  n_train <- nrow(train_subset)
  
  # CASE 1: No data
  if (n_train == 0) {
    preds <- tibble(.pred = rep(0, nrow(test_subset)))
    
    # CASE 2: Small data
  } else if (n_train <= 10) {
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(glmnet_model)
    
    fit_small <- fit(wf, data = train_subset)
    preds <- predict(fit_small, new_data = test_subset)
    
    # CASE 3: Tuned random forest
  } else {
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(rf_model)
    
    grid <- grid_regular(
      mtry(range = c(5, 20)),
      trees(range = c(10, 40)),
      min_n(range = c(2, 10)),
      levels = 4
    )
    
    tuned_results <- tune_grid(
      wf,
      resamples = vfold_cv(train_subset, v = 3),
      grid = grid,
      metrics = metric_set(rmse)
    )
    
    best_params <- select_best(tuned_results, metric = "rmse")
    final_wf <- finalize_workflow(wf, best_params)
    fit_big <- fit(final_wf, data = train_subset)
    preds <- predict(fit_big, new_data = test_subset)
  }
  
  preds <- test_subset %>%
    mutate(Id = paste(Store, Dept, Date, sep = "_")) %>%
    bind_cols(preds) %>%
    select(Id, Weekly_Sales = .pred)
  
  all_predictions[[i]] <- preds
} 

submission <- bind_rows(all_predictions)

vroom_write(submission, file = "./Walmartpreds.csv", delim = ",")

stopCluster(cl)

collect_metrics(tuned_results)
show_best(tuned_results, metric = "rmse", n = 1)

