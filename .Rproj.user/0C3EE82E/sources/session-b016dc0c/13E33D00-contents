# GGG Project
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed) # for target encoding
library(lme4)
library(kknn)
library(doParallel)
library(future)
library(doFuture)

trainData <- vroom("train.csv") %>%
  mutate(type=as.factor(type))
testData <- vroom("test.csv")
glimpse(trainData)

# Detect physical cores
n_physical_cores <- parallel::detectCores(logical = FALSE)  # returns 4 for your machine

# Set up parallel processing using physical cores
plan(multisession, workers = n_physical_cores-1)
registerDoFuture()

my_recipe <- recipe(type~., data=trainData) %>%
  step_normalize(all_numeric_predictors()) #%>% #YES sam no Jonah

library(kernlab)
set.seed(123)
## SVM models
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmPoly)

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmRadial_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmLinear)
#Grid for all
grid_linear <- grid_regular(cost(range = c(-2, 2)), levels = 5)
grid_poly <- grid_regular(
  degree(range = c(1, 3)),
  cost(range = c(-2, 2)),
  levels = 4)
grid_rbf <- grid_regular(
  rbf_sigma(range = c(-5, 5)),
  cost(range = c(-10, 10)),
  levels = 7)
#Folds
set.seed(123)
folds <- vfold_cv(trainData, v = 10, repeats = 2)
#Tuned; Switch wf argument for each
set.seed(123)
tune_linear <- tune_grid(
  svmLinear_wf,
  resamples = folds,
  grid = grid_linear,
  metrics = metric_set(roc_auc))
tune_poly <- tune_grid(
  svmPoly_wf,
  resamples = folds,
  grid = grid_poly,
  metrics = metric_set(roc_auc))
tune_rbf <- tune_grid(
  svmRadial_wf,
  resamples = folds,
  grid = grid_rbf,
  metrics = metric_set(roc_auc))

# extra: best_params <- select_best(svm_tuned, metric = "roc_auc")

svm_results <- bind_rows(
  tune_linear %>% collect_metrics() %>% mutate(model = "Linear"),
  tune_poly %>% collect_metrics() %>% mutate(model = "Polynomial"),
  tune_rbf %>% collect_metrics() %>% mutate(model = "Radial"))

library(ggplot2)
svm_results %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = model, y = mean, fill = model)) +
  geom_boxplot() +
  labs(
    title = "SVM Model Comparison (ROC AUC)",
    y = "Mean ROC AUC",
    x = "Kernel Type") +
  theme_minimal() +
  theme(legend.position = "none")

best_rbf <- select_best(tune_rbf, metric = "roc_auc")

final_wf <- finalize_workflow(svmRadial_wf, best_rbf)

final_fit <- fit(final_wf, data = trainData)

svm_predictions <- predict(final_fit, new_data = testData, type = "class")

kaggle_submission <- svm_predictions %>%
  bind_cols(testData) %>%
  select(id, .pred_class) %>%
  rename(type = .pred_class)

vroom_write(kaggle_submission, "GGGpreds.csv", delim = ",")
#Linear was first submission!
