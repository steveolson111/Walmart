#Walmart Competition
library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(doParallel)
library(themis)

# Parallel
num_cores <- parallel::detectCores()
cl <- makePSOCKcluster(num_cores - 1)
registerDoParallel(cl)

# Load
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

library(dplyr)

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

# NOTE: some of these step functions are not appropriate to use together
my_recipe <- recipe(Weekly_Sales ~ ., data = trainData) %>%
  step_zv(all_predictors()) %>%
  update_role(Store, Dept, Date, new_role = "ID") %>%  # don't turn into predictors
  step_mutate(Store = factor(Store),
              Dept = factor(Dept)) %>%
  step_date(Date, features = c("year","month","week","dow")) %>%
  #step_holiday(Date, holidays = timeDate::listHolidays("US")) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# Choose a regression model
rf_model <- rand_forest(trees = 10, mtry = 20, min_n = 5) %>%
  set_engine("ranger") %>%
  set_mode("regression")

glmnet_model <- linear_reg(penalty = 0.1, mixture = 0.5) %>% 
  set_engine("glmnet")

all_predictions <- list()

groups <- testData %>% distinct(Store, Dept)

for (i in 1:nrow(groups)) {
  store_i <- groups$Store[i]
  dept_i  <- groups$Dept[i]
  
  #message("Processing Store ", store_i, " Dept ", dept_i)
  
  train_subset <- trainData %>% filter(Store == store_i, Dept == dept_i)
  test_subset  <- testData  %>% filter(Store == store_i, Dept == dept_i)
  
  n_train <- nrow(train_subset)
  
  # 1. If NO PAST DATA → predict zero -----------------------------
  if (n_train == 0) {
    preds <- tibble(.pred = rep(0, nrow(test_subset)))
    
    # 2. SMALL DATA: use penalized regression -------------------------
  } else if (n_train <= 10) {
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(glmnet_model)
    
    fit_small <- fit(wf, data = train_subset)
    preds <- predict(fit_small, new_data = test_subset)
    
    # 3. LARGE DATA: random forest -----------------------------------
  } else {
    wf <- workflow() %>%
      add_recipe(my_recipe) %>%
      add_model(rf_model)
    
    fit_big <- fit(wf, data = train_subset)
    preds <- predict(fit_big, new_data = test_subset)
  }
  
  # Store predictions with ID
  preds <- test_subset %>%
    mutate(Id = paste(Store, Dept, Date, sep = "_")) %>%
    bind_cols(preds) %>%
    select(Id, Weekly_Sales = .pred)
  
  all_predictions[[i]] <- preds
}

submission <- bind_rows(all_predictions)

vroom_write(submission, file = "./Walmartpreds.csv", delim = ",")

stopCluster(cl)

# wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(rf_model)
# 
# final_fit <- fit(wf, data = trainData)
# 
# pred <- predict(final_fit, new_data = testData)
# 
# submission <- testData %>%
#   mutate(Id = paste(Store, Dept, Date, sep = "_")) %>%
#   bind_cols(testData, pred) %>%
#   select(Id, Weekly_Sales = .pred)
# vroom_write(submission, file = "./Walmartpreds.csv", delim = ",")

#stopCluster(cl)
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------



# # 
# predictions <- predict(final_model, new_data = testData)
# kaggle_submission <- testData %>%
#   mutate(id = paste(Store, Dept, Date, sep = "_")) %>%
#   bind_cols(predictions) %>%
#   select(id, Weekly_Sales = .pred)
# vroom_write(kaggle_submission, file = "./Walmartpred.csv", delim = ",")

####################################################################
##################### Reg Log WORKFLOW SECTION ###############################
####################################################################
# 
# logRegModel <- logistic_reg() %>% #Type of model
#   set_engine("glm")
# 
# ## Put into a workflow here
# logReg_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logRegModel)%>%
#   fit(data = trainData)

## Make predictions
# amazon_predictions <- predict(logReg_workflow,
#                               new_data=testData,
#                               type="prob") # "class" or "prob"
# ## with type="prob" amazon_predictions will have 2 columns
## one for Pr(0) and the other for Pr(1)!
## with type="class" it will just have one column (0 or 1)
# 
# kaggle_submission <- amazon_predictions %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
# ## Write out the file
# vroom_write(x=kaggle_submission, file="./Logpreds.csv", delim=",")

####################################################################
##################### Penalized Log Reg ###############################
####################################################################
# 
# 
# my_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
#   set_engine("glmnet")
# 
# amazon_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(my_mod)
# 
# ## Grid of values to tune over
# tuning_grid <- grid_regular(penalty(),
#                             mixture(),
#                             levels = 5) ## L^2 total tuning possibilities
# 
# ## Split data for CV
# folds <- vfold_cv(trainData, v = 2, repeats=1) # go to more K folds later
# 
# ## Run the CV18
# CV_results <- amazon_workflow %>%
#   tune_grid(resamples=folds,
#             grid=tuning_grid,
#             metrics=metric_set(roc_auc)) #f_meas, sens, recall, spec,
# #precision, accuracy
# #Or leave metrics NULL
# 
# ## Find Best Tuning Parameters
# bestTune <- CV_results %>%
#   select_best(metric = "roc_auc")
# 
# ## Finalize the Workflow & fit it
# final_wf <-
#   amazon_workflow %>%
#   finalize_workflow(bestTune) %>%
#   fit(data=trainData)
# 
# ## Predict
# amazon_predictions1 <- final_wf %>%
#   predict(new_data = testData, type="prob")
# 
# kaggle_submission1 <- amazon_predictions1 %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
# ## Write out the file
# vroom_write(x=kaggle_submission1, file="./Logpreds1.csv", delim=",")
# 
# ####################################################################
# ##################### Random Forest ###############################
# ####################################################################
# 
# my_mod <- rand_forest(mtry = tune(),
#                       min_n=tune(),
#                       trees=100 ) %>% # 500 or 1000 consider more trees cause why not?
#   set_engine("ranger") %>%
#   set_mode("classification")
# ## Create a workflow with model & recipe
# forest_workflow <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(my_mod)
# 
# ## Set up grid of tuning values
# library(dials)
# mtry_range <- mtry(range = c(1, 9))
# #min_n_range <- min_n(range = c(2,5))
# 
# forest_grid <- grid_regular(mtry_range,
#                             min_n(),
#                             levels = 5) # how many values to try per parameter
# ## Set up K-fold CV
# set.seed(123)
# folds <- vfold_cv(trainData, v = 5, repeats = 1)
# tuned_results <- tune_grid(
#   forest_workflow,
#   resamples = folds,
#   grid = forest_grid,
#   metrics = metric_set(roc_auc))     # Cohen's Kappa)) ## ADD more metrics here?
# ## Find best tuning parameters
# best_params <- select_best(tuned_results, metric = "roc_auc") # choose just one
# 
# ## Finalize workflow and predict
# final_wf <- finalize_workflow(
#   forest_workflow,
#   best_params)
# 
# final_model <- fit(final_wf, data = trainData)
# 
# forest_predictions <- predict(final_model, new_data = testData, type = "prob")
# # Output
# kaggle_submission <- forest_predictions %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
# # Write out the file
# vroom_write(x=kaggle_submission, file="./forestpreds.csv", delim=",") 

####################################################################
##################### Code for KNN ###############################
####################################################################
## knn model
# knn_model <- nearest_neighbor(neighbors=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kknn")
# 
# knn_wf <- workflow() %>%
# add_recipe(my_recipe) %>%
# add_model(knn_model)
# ## Fit or Tune Model HERE
# knn_grid <- grid_regular(neighbors(range = c(15, 25)), levels = 10)
# 
# set.seed(123)
# folds <- vfold_cv(trainData, v = 5)
# 
# tuned_results <- tune_grid(
#   knn_wf,
#   resamples = folds,
#   grid = knn_grid,
#   metrics = metric_set(roc_auc))
# ## Find best tuning parameters
# best_params <- select_best(tuned_results, metric = "roc_auc")
# 
# ## Finalize workflow and predict
# final_wf <- finalize_workflow(
#   knn_wf,
#   best_params)
# 
# finalknn_model <- fit(final_wf, data = trainData)
# 
# knn_predictions <- predict(finalknn_model, new_data=testData, type="prob")
# 
# kaggle_submission <- knn_predictions %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
#   # Write out the file
#   vroom_write(x=kaggle_submission, file="./knnpreds.csv", delim=",")

####################################################################
##################### Naive Bayes ###############################
####################################################################

# library(tidymodels)
# library(discrim)
# 
# ## nb model
# nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
# set_mode("classification") %>%
# set_engine("naivebayes") # install discrim library for the naiveb
# 
# nb_wf <- workflow() %>%
# add_recipe(my_recipe) %>%
# add_model(nb_model)
# 
# ## Tune smoothness and Laplace here
# nb_grid <- grid_regular(
#   Laplace(range = c(0, 1)), 
#   smoothness(range = c(0, 1)), 
#   levels = 5)
# 
# set.seed(123)
# folds <- vfold_cv(trainData, v = 5)
# 
# tuned_nb <- tune_grid(
#   nb_wf,
#   resamples = folds,
#   grid = nb_grid,
#   metrics = metric_set(roc_auc))
# 
# best_params <- select_best(tuned_nb, metric = "roc_auc")
# 
# final_nb_wf <- finalize_workflow(nb_wf, best_params)
# 
# final_nb_model <- fit(final_nb_wf, data = trainData)
# 
# ## Predict
# nb_predictions <- predict(final_nb_model, new_data = testData, type = "prob")
# 
# kaggle_submission <- nb_predictions %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
#   # Write out the file
#   vroom_write(x=kaggle_submission, file="./nbpreds.csv", delim=",")

####################################################################
##################### #Neural Networks ###############################
####################################################################
# library(nnet)
# 
# 
# nn_recipe <- recipe(ACTION ~ ., data = trainData) %>%
#   update_role(MGR_ID, new_role = "id") %>%
#   step_dummy(all_nominal_predictors()) %>% ## Turn color to factor then dummy encode color
#   step_range(all_numeric_predictors(), min = 0, max = 1) #scale to [0,1]
# 
# prep <- prep(nn_recipe)
# baked <- bake(prep, new_data = trainData)
# 
# nn_model <- mlp(hidden_units = tune(),
#                 epochs = 50) %>% #or 100 or 250
# 
# set_engine("nnet") %>% #verbose = 0 prints off less
#   set_mode("classification")
# 
# nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 20)),
#                             levels=10)
# nn_wf <- workflow() %>%
#   add_model(nn_model) %>%
#   add_recipe(nn_recipe)
# 
# set.seed(123)
# folds <- vfold_cv(trainData, v = 5)
# 
# tuned_nn <- nn_wf %>%
#   tune_grid(
#     resamples = folds,
#     grid = nn_tuneGrid,
#     metrics = metric_set(roc_auc))
# 
# ## CV tune, finalize and predict here and save results
# ## This takes a few min (10 on my laptop) so run it on becker if you want
# #create a plot to show
# nn_metrics <- tuned_nn %>% collect_metrics()
# nn_acc <- nn_metrics %>% 
#   filter(.metric == "roc_auc") #or accuracy?
# library(ggplot2)
# 
# ggplot(nn_acc, aes(x = hidden_units, y = mean)) +
#   geom_line(color = "blue", linewidth = 1) +
#   geom_point(size = 2) +
#   labs(
#     title = "Neural Net ROC AUC by Hidden Units",
#     x = "Number of Hidden Units",
#     y = "Mean ROC AUC") +
#   theme_minimal()
# best_params <- select_best(tuned_nn, metric = "roc_auc")
# 
# final_nn_wf <- finalize_workflow(nn_wf, best_params)
# 
# final_nn_model <- fit(final_nn_wf, data = trainData)
# 
# ## Predict
# nn_predictions <- predict(final_nn_model, new_data = testData, type = "prob")
# 
# kaggle_submission <- nn_predictions %>%
#   bind_cols(., testData)%>%
#   select(id,.pred_1)%>%
#   rename(ACTION=.pred_1)#Just keep datetime and prediction variables
#   # Write out the file
#   vroom_write(x=kaggle_submission, file="./nnpreds.csv", delim=",")

####################################################################
##################### PCA SECTION ###############################
####################################################################
#just add the pca line of code

####################################################################
##################### SVM Section ###############################
####################################################################
# 
# my_recipe <- recipe(ACTION~., data=trainData) %>%
#   ##remove for forest:
#   step_mutate_at(all_predictors(), fn = factor) %>% # YESturn all numeric features into factors
#   step_other(all_nominal_predictors(), threshold = .2)%>%  #YES sam, no Jonah Use bigger for less features but crank back down to .001 when scores matter again! # combines categorical values that occur <.1% i
#   step_dummy(all_nominal_predictors())%>%# NO dummy variable encoding7
#   ##remove for forest:
#   #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))%>% #YEStarget encoding (must be 2-f
#   step_normalize(all_numeric_predictors()) #%>% #YES sam no Jonah
# #step_pca(all_predictors(), threshold=.8) #Threshold is between 0 and 1 #only for PCA
# # also step_lencode_glm() and step_lencode_bayes()
# 
# library(kernlab)
# ## SVM models
# svmPoly <- svm_poly(degree=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")
# 
# svmPoly_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svmPoly)
# 
# svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")
# 
# svmRadial_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svmRadial)
# 
# svmLinear <- svm_linear(cost=tune()) %>% # set or tune
#   set_mode("classification") %>%
# set_engine("kernlab")
# 
# svmLinear_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(svmLinear)
# #Grid for all
# grid_linear <- grid_regular(cost(range = c(-2, 2)), levels = 5)
# grid_poly <- grid_regular(
#   degree(range = c(1, 3)),
#   cost(range = c(-2, 2)),
#   levels = 4)
# grid_rbf <- grid_regular(
#   rbf_sigma(range = c(-3, 1)),
#   cost(range = c(-2, 2)),
#   levels = 4)
# #Folds
# set.seed(123)
# folds <- vfold_cv(trainData, v = 2)
# #Tuned; Switch wf argument for each
# set.seed(123)
# tune_linear <- tune_grid(
#   svmLinear_wf,
#   resamples = folds,
#   grid = grid_linear,
#   metrics = metric_set(roc_auc))
# tune_poly <- tune_grid(
#   svmPoly_wf,
#   resamples = folds,
#   grid = grid_poly,
#   metrics = metric_set(roc_auc))
# tune_rbf <- tune_grid(
#   svmRadial_wf,
#   resamples = folds,
#   grid = grid_rbf,
#   metrics = metric_set(roc_auc))
# 
# # extra: best_params <- select_best(svm_tuned, metric = "roc_auc")
# 
# svm_results <- bind_rows(
#   tune_linear %>% collect_metrics() %>% mutate(model = "Linear"),
#   tune_poly %>% collect_metrics() %>% mutate(model = "Polynomial"),
#   tune_rbf %>% collect_metrics() %>% mutate(model = "Radial"))
# 
# library(ggplot2)
# svm_results %>%
#   filter(.metric == "roc_auc") %>%
#   ggplot(aes(x = model, y = mean, fill = model)) +
#   geom_boxplot() +
#   labs(
#     title = "SVM Model Comparison (ROC AUC)",
#     y = "Mean ROC AUC",
#     x = "Kernel Type") +
#   theme_minimal() +
#   theme(legend.position = "none")
# 
# best_rbf <- select_best(tune_rbf, metric = "roc_auc")
# 
# final_wf <- finalize_workflow(svmRadial_wf, best_rbf)
# 
# final_fit <- fit(final_wf, data = trainData)
# 
# svm_predictions <- predict(final_fit, new_data = testData, type = "prob")
# 
# kaggle_submission <- svm_predictions %>%
#   bind_cols(testData) %>%
#   select(id, .pred_1) %>%
#   rename(ACTION = .pred_1)
# 
# vroom_write(kaggle_submission, "svm_best.csv", delim = ",")
# #Linear was first submission!

####################################################################
##################### Balancing Data ###############################
####################################################################

# library(themis)  # for step_smote()
# 
# # Example: assume rFormula = ACTION ~ ., myDataset = trainData, K = 5
# my_recipe <- recipe(ACTION ~ ., data = trainData) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_dummy(all_nominal_predictors()) %>%   # SMOTE requires numeric predictors
#   step_smote(all_outcomes(), neighbors = 2)  # neighbors = K, set e.g. 5
# 
# # Prep the recipe
# prepped_recipe <- prep(my_recipe)
# 
# # Apply to data (bake)
# baked <- bake(prepped_recipe, new_data = trainData)
# 
# 

####################################################################
##################### basic commands for WSL ###############################
####################################################################
# sftp://stat-u03.byu.edu               user is steve111 pass is ssh 
# R CMD BATCH --no-save --no-restore AmazonAnalysis.R &
# top to see top of server 
#q to edit

####################################################################
##################### Ask chat for progress bar code for WSL ###############################
####################################################################

####################################################################
##################### Write code to ask it to save output? ###############################
####################################################################
#vroom_write(...)
#Could use this code in order to save the workflow in order to
#use in r if I had built the workflow in linux.

#save(file="./MyFile.RData", list=c("object1", "object2",...))

#library(doParallel)

#parallel::detectCores() #How many cores do I have?
#cl <- makePSOCKcluster(num_cores)
#registerDoParallel(cl)

#... ## Code here
#stopCluster(cl)

####################################################################
##################### KAGGLE SUBMISSION we want .885 ###############################
####################################################################


# kaggle_submission <- amazon_predictions %>%
# bind_cols(., testData)%>% 
# select(id,.pred_1)%>%
# rename(ACTION=.pred_1)#Just keep datetime and prediction variables
# # Write out the file
# vroom_write(x=kaggle_submission, file="./Logpreds.csv", delim=",")
# predictions <- predict(final_model, new_data = testData)
# kaggle_submission <- testData %>%
#   mutate(id = paste(Store, Dept, Date, sep = "_")) %>%
#   bind_cols(predictions) %>%
#   select(id, Weekly_Sales = .pred)
# vroom_write(kaggle_submission, file = "./Walmartpreds.csv", delim = ",")
# stopCluster(cl)