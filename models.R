library(MASS)
library(readr)
library(readxl)
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(tidytable)
library(yardstick)
library(magrittr)
library(keras)
set.seed(200)

# =======================================================
#                STEP 1: DATA READING
# =======================================================

# secondary cause of condition code
secondary_cause <- read_csv("./data/MIMIC_diagnoses.csv")
secondary_cause_count <- secondary_cause %>% 
  group_by(SUBJECT_ID) %>% 
  count() %>% 
  rename(SECONDARY_DISEASE_COUNT = n) %>% 
  as_tibble()

# process training set
mimic_tr_x <- read_csv("./data/mimic_train_X.csv")
mimic_tr_response <- read_csv("./data/mimic_train_y.csv") # we are predicting this
mimic_tr <- full_join(mimic_tr_x, mimic_tr_response, by = "icustay_id") %>% select(-c(...1.y, ...1.x))

# split into training and testing 
splits <- initial_split(mimic_tr, prop = 0.8, strata = HOSPITAL_EXPIRE_FLAG)
tr_train <- training(splits)
tr_test <- testing(splits)

# process testing set
mimic_ts <- read_csv("./data/mimic_test_X.csv") %>%  select(-...1)

# translate code to main cause of condition  
code_to_main_cause <- read_csv("./data/MIMIC_metadata_diagnose.csv")

# target encode icd9 for training
diagnosis_encoding <- tr_train %>% 
  group_by(ICD9_diagnosis) %>% 
  summarise(icd9_encoded = mean(HOSPITAL_EXPIRE_FLAG, na.rm = TRUE))

global_mean <- mean(tr_train$HOSPITAL_EXPIRE_FLAG, na.rm = TRUE)

# secondary causes
pivot_icd9 <- secondary_cause %>% 
  filter(!is.na(ICD9_CODE)) %>% 
  select(-SEQ_NUM) %>%
  distinct(SUBJECT_ID, HADM_ID, ICD9_CODE) %>% 
  mutate(value = 1) %>% 
  pivot_wider(
    id_cols = c(SUBJECT_ID, HADM_ID),
    names_from = ICD9_CODE,
    values_from = value,
    values_fill = 0
  ) %>% 
  as_tibble()

last_icu <- bind_rows(mimic_tr, mimic_ts) %>%
  group_by(subject_id) %>%
  arrange(desc(ADMITTIME)) %>%
  slice(1) %>%
  ungroup() %>%
  mutate(is_last_icu = 1) %>% 
  select(subject_id, ADMITTIME, is_last_icu)

# =======================================================
#                STEP 2: DATA CLEANING
# =======================================================

# cleaning data 
clean_data <- function(dataset, is_tr) {
  avg_age <- dataset %>% 
    mutate(AGE = as.numeric(difftime(ADMITTIME, DOB, units = 'days')) / 365.25) %>% 
    filter(AGE < 100) %>% 
    summarise(avg = mean(AGE)) %>% 
    pull(avg)
  
  dataset <- dataset %>% 
    mutate(MARITAL_STATUS = case_when (
      MARITAL_STATUS == "LIFE PARTNER" ~ "MARRIED",
      MARITAL_STATUS == "WIDOWED" ~ "SINGLE",
      MARITAL_STATUS == "UNKNOWN (DEFAULT)" ~ "SINGLE",
      MARITAL_STATUS == "SEPARATED" ~ "SINGLE",
      MARITAL_STATUS == "DIVORCED" ~ "SINGLE",
      TRUE ~ MARITAL_STATUS
    )) %>% 
    mutate(ETHNICITY = ifelse(ETHNICITY == 'WHITE', 'WHITE', 'NON_WHITE'))
  
  col_id <- dataset %>% 
    select(subject_id, hadm_id, ADMITTIME)
  
  res <- dataset %>% 
    left_join(secondary_cause_count, by = c('subject_id' = 'SUBJECT_ID')) %>% 
    mutate(AGE = as.numeric(difftime(ADMITTIME, DOB, units = 'days')) / 365.25) %>% 
    mutate(AGE = ifelse(AGE > 100, avg_age, AGE)) %>% 
    left_join(diagnosis_encoding, by = 'ICD9_diagnosis') %>% 
    mutate(icd9_encoded = ifelse(is.na(icd9_encoded), global_mean, icd9_encoded)) %>% 
    select(-c(subject_id, hadm_id, icustay_id, DOB, ADMITTIME, Diff, DIAGNOSIS, ICD9_diagnosis)) %>% 
    get_dummies() %>% 
    select(-c(GENDER, ADMISSION_TYPE, INSURANCE, RELIGION, MARITAL_STATUS, ETHNICITY, FIRST_CAREUNIT))
  
  if (is_tr) {
    res$HOSPITAL_EXPIRE_FLAG <- as.factor(res$HOSPITAL_EXPIRE_FLAG) 
  }
  
  res <- res %>%
    mutate(across(where(is.numeric), ~as.numeric(scale(.)))) %>% 
    bind_cols(col_id) %>% 
    left_join(last_icu, by = c("subject_id", "ADMITTIME")) %>% 
    mutate(is_last_icu = replace_na(is_last_icu, 0)) %>% 
    left_join(pivot_icd9, by = c("subject_id" = "SUBJECT_ID", "hadm_id" = "HADM_ID")) %>% 
    select(-c(subject_id, hadm_id, ADMITTIME)) %>% 
    as_tibble() 
  
  res$is_last_icu <- as.factor(res$is_last_icu) 
  
  return(res)
}

# clean data
tr_tr <- clean_data(tr_train, 1)
tr_ts <- clean_data(tr_test, 1)
tr <- clean_data(mimic_tr, 1)
ts <- clean_data(mimic_ts, 0)

# =======================================================
#             MODEL 1: LOGISTIC REGRESSION
# =======================================================

# model spec
log_fit <- logistic_reg() |> 
  set_engine("glm") |> 
  set_mode("classification") |> 
  translate()

# fit model on training set
log_tr_pred <- log_fit %>% 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = tr_tr)

# in sample prediction
log_tr_fit <- augment(log_tr_pred, tr_tr)

log_tr_fit |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |>
  group_by(HOSPITAL_EXPIRE_FLAG) |>
  mutate(cl_acc = n[.pred_class == HOSPITAL_EXPIRE_FLAG]/sum(n)) |>
  pivot_wider(names_from = .pred_class, 
              values_from = n, values_fill=0) |>
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc) %>% 
  as_tibble()

bal_accuracy(log_tr_fit, HOSPITAL_EXPIRE_FLAG, .pred_class)
log_tr_pred_prob <- augment(log_tr_pred, tr_tr, type="prob")
roc_auc(log_tr_pred_prob, HOSPITAL_EXPIRE_FLAG, .pred_0)

# out of sample prediction
log_ts_fit <- augment(log_tr_pred, tr_ts)

log_ts_fit |> count(HOSPITAL_EXPIRE_FLAG, .pred_class) |>
  group_by(HOSPITAL_EXPIRE_FLAG) |>
  mutate(cl_acc = n[.pred_class == HOSPITAL_EXPIRE_FLAG]/sum(n)) |>
  pivot_wider(names_from = .pred_class, 
              values_from = n, values_fill=0) |>
  select(HOSPITAL_EXPIRE_FLAG, `0`, `1`, cl_acc) %>% 
  as_tibble()

bal_accuracy(log_ts_fit, HOSPITAL_EXPIRE_FLAG, .pred_class)
log_ts_pred_prob <- augment(log_tr_pred, tr_ts, type="prob")
roc_auc(log_ts_pred_prob, HOSPITAL_EXPIRE_FLAG, .pred_0)

# =======================================================
#                MODEL 2: RANDOM FOREST
# =======================================================

# model spec
rf_spec <- rand_forest(mtry=84, trees=1000) |>
  set_mode("classification") |>
  set_engine("ranger")

# fit model
fit_rf <- rf_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = tr_tr)

# in sample prediction 
tr_pred <- predict(fit_rf, tr_tr, type='prob') %>% 
  bind_cols(tr_tr) %>% 
  select(HOSPITAL_EXPIRE_FLAG, .pred_0) %>% 
  as_tibble()

roc_auc(tr_pred, HOSPITAL_EXPIRE_FLAG, .pred_0)

# out of sample prediction
tr_pred <- predict(fit_rf, tr_ts, type='prob') %>% 
  bind_cols(tr_ts) %>% 
  select(HOSPITAL_EXPIRE_FLAG, .pred_0) %>% 
  as_tibble()

roc_auc(tr_pred, HOSPITAL_EXPIRE_FLAG, .pred_0)

# predict test set
ts_pred <- predict(fit_rf, ts, type="prob") %>% 
  bind_cols(mimic_ts) %>% 
  select(icustay_id, .pred_1) %>% 
  rename(ID = 'icustay_id', HOSPITAL_EXPIRE_FLAG = '.pred_1') %>% 
  as_tibble()

write_csv(ts_pred, "res.csv")

# =======================================================
#                MODEL 3: BOOSTED TREE
# =======================================================

bt_spec <- boost_tree(
  trees = 1000,             
  tree_depth = 6,
  learn_rate = 1.023293,
  loss_reduction = 1,
  mtry = 83   
) |>
  set_mode("classification") |>
  set_engine("xgboost")

fit_bt <- bt_spec |> 
  fit(HOSPITAL_EXPIRE_FLAG ~ ., data = tr_tr)

# in sample prediction
tr_pred <- predict(fit_bt, tr_tr, type='prob') %>% 
  bind_cols(tr_tr) %>% 
  select(HOSPITAL_EXPIRE_FLAG, .pred_0) %>% 
  as_tibble()

roc_auc(tr_pred, HOSPITAL_EXPIRE_FLAG, .pred_0)

# out of sample prediction
tr_pred <- predict(fit_bt, tr_ts, type='prob') %>% 
  bind_cols(tr_ts) %>% 
  select(HOSPITAL_EXPIRE_FLAG, .pred_0) %>% 
  as_tibble()

roc_auc(tr_pred, HOSPITAL_EXPIRE_FLAG, .pred_0)

# predict test set
ts_pred <- predict(fit_bt, ts, type="prob") %>% 
  bind_cols(mimic_ts) %>% 
  select(icustay_id, .pred_1) %>% 
  rename(ID = 'icustay_id', HOSPITAL_EXPIRE_FLAG = '.pred_1') %>% 
  as_tibble()

write_csv(ts_pred, "res.csv")

# =======================================================
#                MODEL 4: NEURAL NETWORK
# =======================================================

# train
x_train <- tr_tr %>% select(-HOSPITAL_EXPIRE_FLAG) %>% as.matrix() 
y_train <- tr_tr$HOSPITAL_EXPIRE_FLAG %>% as.integer() %>% subtract(1)

# test
x_test <- tr_ts %>% select(-HOSPITAL_EXPIRE_FLAG) %>% as.matrix()

nn_model <- keras_model_sequential() %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(x_train)) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

nn_model |> compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = c("accuracy", metric_auc(name = "AUC"))
)

nn_model |> fit(
  x = x_train,
  y = y_train,
  epochs = 50,
  batch_size = 256,
  validation_split = 0.2,
  verbose = 1
)

# in sample predictions
tr_pred <- predict(nn_model, x_train, verbose = 0) %>% 
  as_tibble() %>% 
  bind_cols(tr_train) %>% 
  select(HOSPITAL_EXPIRE_FLAG, V1) %>% 
  as_tibble()

tr_pred$HOSPITAL_EXPIRE_FLAG <- as.factor(tr_pred$HOSPITAL_EXPIRE_FLAG) 
roc_auc(tr_pred, HOSPITAL_EXPIRE_FLAG, V1)

# out of sample predictions
ts_pred <- predict(nn_model, x_test, verbose = 0) %>% 
  as_tibble() %>% 
  bind_cols(tr_test) %>% 
  select(HOSPITAL_EXPIRE_FLAG, V1) %>% 
  as_tibble()

ts_pred$HOSPITAL_EXPIRE_FLAG <- as.factor(ts_pred$HOSPITAL_EXPIRE_FLAG) 
roc_auc(ts_pred, HOSPITAL_EXPIRE_FLAG, V1)