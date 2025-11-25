###olive oil consumption
### author: Youmin Li
## library
install.packages(c("ISLR2","tidyverse","leaps","glmnet","pls"))
library(ISLR2); library(tidyverse); library(leaps); library(glmnet); library(pls)
library(tidyverse); library(readxl); library(janitor)
library(psych); library(GPArotation); library(tidyverse)
library(NbClust); library(factoextra); library(cluster); library(fpc)
library(seminr); library(tidyverse)
packageVersion("seminr")

# ----------------------------
### data clean
save.image("model_progress.RData")
load("")
dat_seg <- scored_with_clusters
#-------------------------------------------------------------------------------------
###############PLS-SEM Estimation
# ------------------------------------------
# Define blocks for PLS-SEM

blocks <- list(
  taste      = c("taste_1","taste_2","taste_3"),
  price      = c("price_1","price_2","price_3", "price_4"),
  trust_evoo = c("act_evoo_1","act_evoo_2","act_evoo_3","act_evoo_4"),
  trust_roo  = c("act_roo_1","act_roo_2","act_roo_3","act_roo_4"),
  health_neg     = c("neg_con_1","neg_con_3"),
  evoo_cons   = c("evoo_con", "evoo_uses"),
  roo_con    = c("roo_con", "roo_uses")
)

dat_seg <- dat_seg |>
  mutate(
    income = na_if(income, 6),  # convert "not answer" into NA
    income_grp = case_when(
      income %in% 1:2 ~ "Low",
      income %in% 3:5 ~ "High",
      TRUE ~ NA_character_
    ),
    income_grp = factor(income_grp, levels = c("Low","High"))
  )

m <- dat_seg |>
  mutate(
    gender = factor(gender, levels = c("Male","Female")),
    income = na_if(income, 6),
    income_grp = case_when(
      income %in% 1:2 ~ "Low",
      income %in% 3:5 ~ "High",
      TRUE ~ NA_character_
    ),
    income_grp = factor(income_grp, levels = c("Low","High"))
  ) |>
  select(
    all_of(blocks$evoo_cons),
    all_of(blocks$roo_con),
    all_of(blocks$trust_evoo),
    all_of(blocks$trust_roo),
    all_of(blocks$taste),
    all_of(blocks$price),
    all_of(blocks$health_neg),
    neg_con_2,
    gender, income_grp, schooling, and_mad
  ) |>
  drop_na()


# Measurement model (reflective for Trust/Taste; formative Price optional—here reflective for simplicity)
mm <- constructs(
  composite("TASTE",          multi_items("taste_", 1:3), weights = mode_A),
  composite("PRICE",          c("price_1","price_3")),
  composite("TRUST_EVOO",     multi_items("act_evoo_", 1:4), weights = mode_A),
  composite("TRUST_ROO",      multi_items("act_roo_", 1:4),  weights = mode_A),
  composite("HEALTH_NEG",     c("neg_con_1","neg_con_3")),
  composite("EVOO_CONS",      c("evoo_con", "evoo_uses")),
  composite("ROO_CONS",       c("roo_con",  "roo_uses"))
)

sm <- relationships(
  paths(from = c("TASTE","PRICE","HEALTH_NEG"), to = c("TRUST_EVOO","TRUST_ROO")),
  paths(from = c("TASTE","PRICE","HEALTH_NEG","TRUST_EVOO","TRUST_ROO"), to = "EVOO_CONS"),
  paths(from = c("TASTE","PRICE","HEALTH_NEG","TRUST_EVOO","TRUST_ROO"), to = "ROO_CONS")
)

# indicators used in the measurement model
indicators <- c(
  "taste_1","taste_2","taste_3",
  "price_1","price_3",
  "act_evoo_1","act_evoo_2","act_evoo_3","act_evoo_4",
  "act_roo_1","act_roo_2","act_roo_3","act_roo_4",
  "neg_con_1","neg_con_3",
  "evoo_con","evoo_uses",
  "roo_con","roo_uses"
)

# 🔁 IMPORTANT: build from dat_seg, not m, and DO NOT drop_na() yet
m_sem <- dat_seg |>
  dplyr::select(dplyr::all_of(indicators)) |>
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric))

# Structural model
pls_all <- estimate_pls(
  data = m_sem,
  measurement_model = mm,
  structural_model = sm,
  inner_weights = path_weighting
)
summary(pls_all)
boot_all <- bootstrap_model(pls_all, nboot = 5000)
summary(boot_all)
loadings <- pls_all$outer_loadings
loadings


# Multi-group by gender / income / region============================
bootstrap_model(estimate_pls(d, mm, sm), nboot = 3000, verbose = FALSE, parallel = FALSE)

fit_mga <- function(split_var) {
  groups <- split(m_sem, m_sem[[split_var]])
  ests  <- lapply(groups, function(d) {
    model <- estimate_pls(d, mm, sm)
    bootstrap_model(model, nboot = 3000, parallel = FALSE, verbose = FALSE)
  })
  list(groups = names(groups), boots = ests)
}

mga_gender <- fit_mga("gender")
mga_income <- fit_mga("income_grp")
mga_region <- fit_mga("region")

saveRDS(list(pls_all=pls_all, boot_all=boot_all,
             mga_gender=mga_gender, mga_income=mga_income, mga_region=mga_region),
        "outputs/sem_pls_objects.rds")

###-====================================================================================
# K-fold CV (10x, repeat 5 times)
set.seed(1)
pred <- predict_pls(
  model    = pls_all,
  technique= predict_DA,   # direct antecedents (common choice)
  noFolds  = 10,           # k-fold
  reps     = 5             # repeats for stability
)

# Out-of-sample residuals by indicator: PLS vs LM benchmark
rmse_pls <- sqrt(colMeans(pred$PLS_out_of_sample_residuals^2, na.rm = TRUE))
rmse_lm  <- sqrt(colMeans(pred$lm_out_of_sample_residuals^2,  na.rm = TRUE))
summary_tbl <- data.frame(
  indicator = names(rmse_pls),
  RMSE_PLS  = rmse_pls,
  RMSE_LM   = rmse_lm,
  ΔRMSE     = rmse_pls - rmse_lm
)
summary_tbl[order(summary_tbl$ΔRMSE), ]

# Quick rule: if most ΔRMSE < 0, your model shows predictive power.
# For small N, use LOOCV instead:
pred_loocv <- predict_pls(pls_all, technique = predict_DA, noFolds = NULL)  # LOOCV

# matrix or formula interface; 10-fold CV by default
fit <- plsr(evoo_cons ~ ., data = ml, scale = TRUE, validation = "CV", segments = 10)

# RMSEP across components & the "one-SE" pick
plot(RMSEP(fit), legendpos = "topright")
best_ncomp <- selectNcomp(fit, method = "onesigma", plot = TRUE)
best_ncomp

library(caret)
set.seed(42)
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
pls_cv <- train(
  evoo_cons ~ ., data = ml,
  method = "pls",
  tuneLength = 20,
  trControl = ctrl,
  preProcess = c("center","scale"),
  metric = "RMSE"
)
pls_cv$bestTune    # chosen ncomp
pls_cv$results     # RMSE/MAE across ncomp


#-------------------------------------------------------
### LASSO+AIC/BIC selection
library(glmnet); library(rsample); library(yardstick); library(vip); library(MASS)

# Design matrix for EVOO consumption
ml <- dat_seg |> drop_na(evoo_cons) |>
  transmute(
    evoo_cons,
    taste1=taste_1, taste2=taste_2, taste3=taste_3,
    price1=price_1, price2=price_2, price3=price_3,
    trustE1=trust_evoo1,trustE2=trust_evoo2,trustE3=trust_evoo3,trustE4=trust_evoo4,
    trustR1=trust_roo1,trustR2=trust_roo2,trustR3=trust_roo3,trustR4=trust_roo4,
    health=health,
    gender=as.factor(gender), income=as.factor(income_grp), region=as.factor(region),
    segment=as.factor(segment)
  )

x <- model.matrix(evoo_cons ~ . , ml)[,-1]
y <- ml$evoo_cons

set.seed(42)
cv <- cv.glmnet(x, y, alpha = 1, nfolds = 10, family = "gaussian", standardize = TRUE)
fit_lasso <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min, standardize = TRUE)

# Variable importance & performance
vi <- broom::tidy(fit_lasso) |> filter(lambda == min(lambda))
pred <- predict(fit_lasso, newx = x)
rmse <- yardstick::rmse_vec(truth = y, estimate = as.numeric(pred))

saveRDS(list(cv=cv, fit=fit_lasso, vi=vi, rmse=rmse), "outputs/lasso_evoo.rds")

# AIC/BIC stepwise around a GLM (transparent selection)
base_glm <- lm(evoo_cons ~ taste_1 + taste_2 + taste_3 + price_1 + price_2 + price_3 +
                 trust_evoo1 + trust_evoo2 + trust_evoo3 + trust_evoo4 +
                 trust_roo1 + trust_roo2 + trust_roo3 + trust_roo4 +
                 health + gender + income_grp + region + segment, data = dat_seg)
step_aic <- stepAIC(base_glm, direction = "both", trace = FALSE, k = 2)       # AIC
step_bic <- stepAIC(base_glm, direction = "both", trace = FALSE, k = log(nrow(dat_seg)))  # BIC
saveRDS(list(aic=step_aic, bic=step_bic), "outputs/ic_models.rds")



# Best subset via regsubsets
regfit.full <- regsubsets(Salary ~ ., data = hit_tr, nvmax = ncol(hit_tr)-1)
summary.full <- summary(regfit.full)

# Choose by BIC
which.min(summary.full$bic)
coef(regfit.full, which.min(summary.full$bic))

# 10-fold CV to choose model size
K <- 10
folds <- sample(rep(1:K, length.out = nrow(hit_tr)))
cv.err <- matrix(NA, K, 19)
xcols <- setdiff(names(hit_tr), "Salary")

# helper to predict from regsubsets
predict.regsubsets <- function(object, newdata, id) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id = id)
  xvars <- names(coefi)
  as.numeric(mat[, xvars] %*% coefi)
}

for (k in 1:K) {
  fit.k <- regsubsets(Salary ~ ., data = hit_tr[folds != k,], nvmax = 19)
  for (m in 1:19) {
    yhat <- predict.regsubsets(fit.k, hit_tr[folds == k,], id = m)
    cv.err[k, m] <- mean((hit_tr$Salary[folds == k] - yhat)^2)
  }
}
m.opt <- which.min(colMeans(cv.err))
m.opt
coef(regfit.full, m.opt)

# Test MSE for chosen size
yhat.te <- predict.regsubsets(regfit.full, hit_te, id = m.opt)
mean((hit_te$Salary - yhat.te)^2)

# PCR
set.seed(1)
pcr.fit <- pcr(Salary ~ ., data = hit_tr, scale = TRUE, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")
# Choose #components by CV
comp.opt.pcr <- which.min(pcr.fit$validation$PRESS)
comp.opt.pcr
pcr.pred <- predict(pcr.fit, hit_te, ncomp = comp.opt.pcr)
mean((hit_te$Salary - pcr.pred)^2)

# PLS
set.seed(1)
pls.fit <- plsr(Salary ~ ., data = hit_tr, scale = TRUE, validation = "CV")
validationplot(pls.fit, val.type = "MSEP")
comp.opt.pls <- which.min(pls.fit$validation$PRESS)
comp.opt.pls
pls.pred <- predict(pls.fit, hit_te, ncomp = comp.opt.pls)
mean((hit_te$Salary - pls.pred)^2)


# -----------------------------
####### Ridge & Lasso
# -----------------------------
x_tr <- model.matrix(Salary ~ ., hit_tr)[, -1]
y_tr <- hit_tr$Salary
x_te <- model.matrix(Salary ~ ., hit_te)[, -1]
y_te <- hit_te$Salary

# Ridge (alpha=0), lambda grid
grid <- 10^seq(5, -2, length=200)
ridge.cv <- cv.glmnet(x_tr, y_tr, alpha = 0, lambda = grid, standardize = TRUE)
ridge.cv$lambda.min; ridge.cv$lambda.1se
ridge.fit <- glmnet(x_tr, y_tr, alpha=0, lambda = ridge.cv$lambda.min, standardize = TRUE)
coef(ridge.fit)
pred.ridge <- predict(ridge.fit, s = ridge.cv$lambda.min, newx = x_te)
mean((y_te - pred.ridge)^2)

# Lasso (alpha=1)
lasso.cv <- cv.glmnet(x_tr, y_tr, alpha = 1, lambda = grid, standardize = TRUE)
lasso.cv$lambda.min; lasso.cv$lambda.1se
lasso.fit <- glmnet(x_tr, y_tr, alpha=1, lambda = lasso.cv$lambda.min, standardize = TRUE)
coef(lasso.fit)
pred.lasso <- predict(lasso.fit, s = lasso.cv$lambda.min, newx = x_te)
mean((y_te - pred.lasso)^2)

### model compare (non-linear vs linear)
anova(model1, model2)


# R/99_helpers.R
ci_fmt <- function(x) sprintf("%.3f", x)
plot_coef_ci <- function(boot_obj, path, file){
  library(dplyr); library(ggplot2)
  coefs <- boot_obj$boot_paths |> 
    filter(path %in% path) |>
    group_by(path) |>
    summarise(beta = mean(original), lo = quantile(boot_means, .025), hi = quantile(boot_means, .975))
  gg <- ggplot(coefs, aes(x=path, y=beta)) +
    geom_point() + geom_errorbar(aes(ymin=lo, ymax=hi), width=.2) +
    coord_flip() + theme_minimal() + labs(y="Path coefficient", x="")
  ggsave(file, gg, width=6, height=4, dpi=300)
}

#####---------------------------------------------------------------------
# prediction - descision tree for olive oil consumption choice --- to chek cluster-pred.r