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

# Load data
trainData <- vroom("train.csv")
testData  <- vroom("test.csv")
features <- vroom("features.csv")
stores <- vroom("stores.csv")

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

# Recipe
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

######## GLOBAL CV TUNING ########
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
  trees(range = c(10, 40)), # small for speed
  min_n(range = c(2, 10)),
  levels = 4
)

tuned_results_full <- tune_grid(
  wf_full,
  resamples = folds_full,
  grid = grid_full,
  metrics = metric_set(rmse)
)

# Best hyperparameters
best_params_full <- select_best(tuned_results_full, metric = "rmse")
print(best_params_full)

# Cross-validated RMSE for LearningSuite
cv_rmse_to_report <- tuned_results_full %>%
  collect_metrics() %>%
  filter(.config == best_params_full$.config)
print(cv_rmse_to_report)

########################
# Store/Dept Predictions
########################
glmnet_model <- linear_reg(penalty = 0.1, mixture = 0.5) %>% 
  set_engine("glmnet")

all_predictions <- list()
groups <- testData %>% distinct(Store, Dept)

for (i in 1:nrow(groups)) {
  store_i <- groups$Store[i]
  dept_i  <- groups$Dept[i]
  
  train_subset <- trainData %>% filter(Store == store_i, Dept == dept_i)
  test_subset  <- testData  %>% filter(Store == store_i, Dept == dept_i)
  
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
    
    # CASE 3: Use globally tuned RF
  } else {
    rf_best <- rand_forest(
      trees = best_params_full$trees,
      mtry = best_params_full$mtry,
      min_n = best_params_full$min_n
    ) %>%
      set_engine("ranger") %>%
      set_mode("regression")
    
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(rf_best)
    
    fit_big <- fit(wf, data = train_subset)
    preds <- predict(fit_big, new_data = test_subset)
  }
  
  preds <- test_subset %>%
    mutate(Id = paste(Store, Dept, Date, sep = "_")) %>%
    bind_cols(preds) %>%
    select(Id, Weekly_Sales = .pred)
  
  all_predictions[[i]] <- preds
}

submission <- bind_rows(all_predictions)
vroom_write(submission, file = "./Walmartpred.csv", delim = ",")

stopCluster(cl)
