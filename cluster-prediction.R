# =========================================================
# Olive oil ML pipeline: EFA/PCA + Clustering + Models + ROC
# Author: Youmin Li
# =========================================================
# ---- 0) Packages ----
pkgs <- c(
  "tidyverse","readxl","janitor","stringr",
  "psych","EFAtools",
  "cluster","factoextra","NbClust",
  "tidymodels","vip","glmnet","ranger","xgboost","mgcv",
  "pROC","ggrepel"
)
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE)
invisible(lapply(pkgs, library, character.only = TRUE))
library(tidyverse); library(readxl); library(janitor)
set.seed(1234)
# ---- 1) Load data ----
# Change this path if you run locally
save(dat_model, rec_cls, folds, res_lasso, res_rf, res_xgb,
     file = "progress_models.RData")
load("progress_models.RData")

DATA_PATH <- "/Users/macpro/Desktop/Youmin-phd/Machine learning/Project/Base_v1.xlsx"
dat0 <- readxl::read_xlsx(DATA_PATH) %>% janitor::clean_names()

# ---- 2) Select perception items (edit patterns to your headers) ----
percept_patterns <- c("^act_evoo_", "^act_roo_", "^taste_", "price_1","price_3", "neg_con_1", "neg_con_3")
percept_cols <- names(dat0)[Reduce(`|`, lapply(percept_patterns, function(p)
  str_detect(names(dat0), regex(p, ignore_case = TRUE))))]

stopifnot(length(percept_cols) > 3)  # need multiple items

#reverse-code known reverse items (1–7 Likert example). 
# rev_items <- intersect(c("taste_1","taste_2","taste_3","health_2","health_3"), names(df0))
# df0 <- df0 %>%
#   mutate(across(all_of(rev_items), ~ ifelse(is.na(.x), NA_real_, 8 - as.numeric(.x))))

X_raw <- dat0 %>% select(all_of(percept_cols)) %>% mutate(across(everything(), as.numeric))
row_ok <- rowMeans(is.na(X_raw)) <= 0.20
X <- X_raw[row_ok, , drop = FALSE]

# Impute item-level missing with item median to run diagnostics; EFA will use complete cases after scaling
X_imp <- X %>% mutate(across(everything(), ~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x)))

# ---- 3) EFA / PCA diagnostics (KMO, Bartlett) ----
R <- cor(X_imp, use = "pairwise.complete.obs")
kmo <- psych::KMO(R)     # overall MSA + item MSAs
bart <- psych::cortest.bartlett(R, n = nrow(X_imp))
sink("TAB_KMO_Bartlett.txt")
cat("KMO overall MSA:", round(kmo$MSA, 4), "\n")
print(kmo$MSAi)
cat("\nBartlett's test:\n")
print(bart)
sink()

# ---- 4) Determine # of factors (Parallel analysis + Scree) ----
X_scaled <- scale(X_imp)
pdf("FIG_ScreePlot_EFA.pdf", width = 6, height = 4)
fa_par <- psych::fa.parallel(X_scaled, fa = "fa", fm = "ml", show.legend = FALSE)
dev.off()
nf <- fa_par$nfact; if (is.null(nf) || is.na(nf)) nf <- 3L
message(sprintf("Selected number of factors: %s", nf))

# ---- 5) EFA with ML & oblimin; output loadings + factor scores ----
fa_fit <- psych::fa(X_scaled, nfactors = nf, fm = "ml", rotate = "oblimin", scores = "regression")
# Loadings table
load_tab <- tibble(variable = rownames(fa_fit$loadings[])) %>%
  bind_cols(as_tibble(unclass(fa_fit$loadings))) %>%
  arrange(variable)
readr::write_csv(load_tab, "TAB_FactorLoadings.csv")

# Factor scores (only for rows we used)
fa_scores <- as_tibble(fa_fit$scores) %>% setNames(paste0("F", seq_len(ncol(.))))
row_ids <- which(row_ok)  # map back to original df0 rows
score_out <- dat0[row_ids, , drop = FALSE] %>% 
  mutate(.row_id = row_ids) %>% 
  bind_cols(fa_scores)
readr::write_csv(score_out %>% select(.row_id, starts_with("F")), "TAB_FactorScores.csv")

# ---- 6) K-means clustering on factor scores; pick k by silhouette ----
Fmat <- as.matrix(fa_scores)
sil_df <- map_df(2:10, function(k) {
  km <- kmeans(Fmat, centers = k, nstart = 50)
  ss <- silhouette(km$cluster, dist(Fmat))
  tibble(k = k, mean_sil = mean(ss[, "sil_width"]))
})
k_opt <- sil_df$k[which.max(sil_df$mean_sil)]
message(sprintf("Optimal k by silhouette: %s", k_opt))

ggplot(sil_df, aes(k, mean_sil)) +
  geom_line() + geom_point() +
  labs(x = "k", y = "Mean silhouette width", title = "Silhouette selection for K-means")

km_opt <- kmeans(Fmat, centers = k_opt, nstart = 100)
cluster_labels <- factor(km_opt$cluster)

# Attach clusters to factor scores
scored_with_clusters <- score_out %>% mutate(cluster_k = cluster_labels)
scored_with_clusters$evoo_choice <- dat$prefers_evoo
readr::write_csv(scored_with_clusters, "scored_with_clusters.csv")

# ---- 7) Visualize clusters on first two PCs of factor scores ----
pc_fac <- prcomp(Fmat, scale. = TRUE)
pc_df <- as_tibble(pc_fac$x[, 1:2]) %>%
  mutate(cluster_k = cluster_labels)
ggplot(pc_df, aes(PC1, PC2, color = cluster_k)) +
  geom_point(alpha = 0.75) +
  labs(title = "Clusters projected on PCs of factor scores")

# ---- 8) Export cluster sizes ----
tab_sizes <- scored_with_clusters %>%
  count(cluster_k, name = "n") %>%
  mutate(prop = n / sum(n))
readr::write_csv(tab_sizes, "TAB_ClusterSizes.csv")

# ---- 9) Quick cluster profiles (means of factors + a few covariates if present) ----
# Pick a few useful covariates if they exist
covars <- intersect(c("age_cont","age","age_group","gender","income","education","region",
                      "evoo_uses","roo_uses","evoo_con","roo_con"), names(scored_with_clusters))
prof <- scored_with_clusters %>%
  select(cluster_k, starts_with("F"), all_of(covars)) %>%
  group_by(cluster_k) %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE), .names = "mean_{.col}"),
            .groups = "drop")
readr::write_csv(prof, "TAB_ClusterProfiles.csv")

message("Stage I complete. Files written:\n",
        "  - FIG_ScreePlot_EFA.pdf\n",
        "  - TAB_KMO_Bartlett.txt\n",
        "  - TAB_FactorLoadings.csv\n",
        "  - TAB_FactorScores.csv\n",
        "  - FIG_Silhouette_kmeans.pdf\n",
        "  - FIG_Clusters_on_FactorPCs.pdf\n",
        "  - TAB_ClusterSizes.csv\n",
        "  - TAB_ClusterProfiles.csv")

### cluster 2 choose EVOO
### cluster 1 chosse ROO
#------------5 segments cluster -  price sensitive-tast driven-healthy perception (1,2,3)---------------
# Identify factor columns (those starting with F)
Fcols <- grep("^F\\d+$", names(scored_with_clusters), value = TRUE)
Fmat  <- scale(scored_with_clusters[Fcols])    # scale factors before clustering
###Decide how many clusters (based on perceptions)
# Try 3–6 to detect price/taste/health segments
fviz_nbclust(Fmat, kmeans, method = "silhouette") +
  labs(title = "Silhouette for perception-based clusters")

k_new <- 5    
set.seed(1234)
km_new <- kmeans(Fmat, centers = k_new, nstart = 100)

scored_with_clusters <- scored_with_clusters %>%
  mutate(percept_cluster = factor(km_new$cluster))

# Examine mean factor profiles for each new cluster
cluster_profiles <- scored_with_clusters %>%
  group_by(percept_cluster) %>%
  summarise(across(all_of(Fcols), mean, na.rm = TRUE)) %>%
  arrange(percept_cluster)

print(cluster_profiles)

# ==========================================================
# Label clusters manually based on dominant factors
# ==========================================================
# After checking which factors are highest per cluster,change based on your factor meanings
labels <- c(
  "1" = "Price sensitive",
  "2" = "Healthy perception",
  "3" = "EVOO favor",
  "4" = "Tast driven",
  "5" = "ROO favor"
)
scored_with_clusters <- scored_with_clusters %>%
  mutate(
    segment = percept_cluster,                          # numeric cluster ID
    segment_label = labels[as.character(segment)]        # descriptive label
  )

#  Visualize perception-based clusters
# ==========================================================
pc_fac <- prcomp(Fmat, scale. = TRUE)
pc_df <- as_tibble(pc_fac$x[, 1:2]) %>%
  mutate(
    segment = scored_with_clusters$segment,
    segment_label = scored_with_clusters$segment_label
  )

ggplot(pc_df, aes(PC1, PC2, color = segment_label)) +
  geom_point(size = 2.2, alpha = 0.85) +
  labs(
    title = "Perception-Based Consumer Segments (PC Projection)",
    color = "Segment"
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    legend.position = "right"
  )

# Export new cluster results
write_csv(scored_with_clusters, "scored_with_newsegments.csv")
write_csv(cluster_profiles, "TAB_NewSegmentProfiles.csv")
save(scored_with_clusters, file = "my_objects.RData")

# ---- 5) Tidymodels setup ----
set.seed(1234)
scored_with_clusters$evoo_choice <- dat$prefers_evoo
dat_model <- scored_with_clusters %>%
  select(
    evoo_choice,
    starts_with("F"),
    all_of(demo_cols)
  ) %>%
  filter(!is.na(evoo_choice)) %>%
  mutate(evoo_choice = factor(evoo_choice))

rec_cls <- recipe(
  evoo_choice ~ ., 
  data = dat_model
) %>%
  step_zv(all_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_normalize(all_numeric_predictors())

# Stratified 10-fold CV on binary outcome
folds <- vfold_cv(dat_model, v = 10, strata = evoo_choice)
metric_set_cls <- metric_set(roc_auc, accuracy, mn_log_loss)

# ---- 6) Models ----
# 6.1 Logistic baseline
mod_logit <- logistic_reg(mode = "classification") %>% set_engine("glm")
wf_logit <- workflow() %>% add_model(mod_logit) %>% add_recipe(rec_cls)
res_logit <- fit_resamples(wf_logit, folds, metrics = metric_set_cls, control = control_resamples(save_pred = TRUE))
best_logit <- select_best(res_logit, metric = "roc_auc")
final_logit <- finalize_workflow(wf_logit, best_logit)

# 6.2 LASSO logistic
mod_lasso <- logistic_reg(mode="classification", penalty = tune(), mixture = 1) %>% set_engine("glmnet")
wf_lasso <- workflow() %>%
  add_recipe(rec_cls) %>%     # <-- correct
  add_model(mod_lasso)
grid_lasso <- grid_regular(
  penalty(range = c(-4, 0)),
  levels = 30
)

res_lasso <- tune_grid(
  wf_lasso,
  resamples = folds,
  grid = grid_lasso,
  metrics = metric_set_cls
)
collect_metrics(res_lasso)

best_lasso <- select_best(res_lasso, metric = "roc_auc")
final_lasso <- finalize_workflow(wf_lasso, best_lasso)
fit_lasso <- fit_resamples(
  final_lasso,
  folds,
  metrics = metric_set_cls,
  control = control_resamples(save_pred = TRUE)
)

# 6.3 GAM (logistic)
factor_cols <- names(dat_model_clean)[grepl("^F\\d+$", names(dat_model_clean))]
nf <- min(4, length(factor_cols))  # number of smooth terms
smooth_terms <- paste0("s(", factor_cols[1:nf], ")", collapse = " + ")
gam_formula <- as.formula(
  paste("evoo_choice ~", smooth_terms)
)
# Use parsnip mgcv GAM via formula smoothing; choose a couple of likely nonlinear factors
mod_gam <- gen_additive_mod(
  mode = "classification"
) %>% 
  set_engine("mgcv", family = binomial())

# Note: using a custom workflow via formula
wf_gam <- workflow() %>%
  add_formula(gam_formula) %>%
  add_model(mod_gam)

res_gam <- fit_resamples(
  wf_gam,
  folds,
  metrics = metric_set_cls,
  control = control_resamples(save_pred = TRUE)
)

# 6.4 Random Forest
mod_rf <- rand_forest(mode="classification", mtry = tune(), trees = 800, min_n = tune()) %>% set_engine("ranger", importance = "permutation")
wf_rf <- workflow() %>% add_model(mod_rf) %>% add_recipe(rec_cls)
grid_rf <- grid_regular(mtry(range = c(2L, length(predictor_cols))), min_n(), levels = 6)
res_rf <- tune_grid(wf_rf, resamples = folds, grid = grid_rf, metrics = metric_set_cls)
best_rf <- select_best(res_rf, metric = "roc_auc")
final_rf <- finalize_workflow(wf_rf, best_rf)
fit_rf <- fit_resamples(final_rf, folds, metrics = metric_set_cls, control = control_resamples(save_pred = TRUE))

# 6.5 Boosting (XGBoost)
mod_xgb <- boost_tree(mode="classification",
                      trees = tune(), learn_rate = tune(), mtry = tune(),
                      tree_depth = tune(), loss_reduction = tune(), min_n = tune()) %>%
  set_engine("xgboost")
wf_xgb <- workflow() %>% add_model(mod_xgb) %>% add_recipe(rec_cls)
grid_xgb <- grid_latin_hypercube(
  trees(), learn_rate(), mtry(range = c(2L, length(predictor_cols))),
  tree_depth(), loss_reduction(), min_n(),
  size = 20
)
res_xgb <- tune_grid(wf_xgb, resamples = folds, grid = grid_xgb, metrics = metric_set_cls)
best_xgb <- select_best(res_xgb, metric = "roc_auc")
final_xgb <- finalize_workflow(wf_xgb, best_xgb)
fit_xgb <- fit_resamples(final_xgb, folds, metrics = metric_set_cls, control = control_resamples(save_pred = TRUE))

# ---- 7) Summaries: CV metrics table ----
tab_metrics <- bind_rows(
  collect_metrics(res_logit) %>% mutate(model = "Logistic"),
  collect_metrics(fit_lasso) %>% mutate(model = "LASSO-logit"),
  collect_metrics(fit_rf) %>% mutate(model = "Random Forest"),
  collect_metrics(fit_xgb) %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, mean, std_err) %>%
  pivot_wider(names_from = .metric, values_from = c(mean, std_err)) %>%
  arrange(desc(mean_roc_auc))

write_csv(tab_metrics, "TAB_CV_Metrics_Classification.csv")

# ---- 8) ROC curves (combined) ----
preds_all <- bind_rows(
  collect_predictions(res_logit)  %>% mutate(model = "Logistic"),
  collect_predictions(fit_lasso)  %>% mutate(model = "LASSO-logit"),
  collect_predictions(fit_rf)     %>% mutate(model = "Random Forest"),
  collect_predictions(fit_xgb)    %>% mutate(model = "XGBoost")
)

roc_df <- preds_all %>%
  group_by(model, id) %>%
  roc_curve(truth = !!sym(OUTCOME_BIN), .pred_1) %>%
  ungroup()

autoplot_roc <- roc_df %>%
  ggplot(aes(1 - specificity, sensitivity, color = model)) +
  geom_path(size = 0.9) + geom_abline(linetype = 2) +
  coord_equal() +
  labs(title = "ROC curves (10-fold CV)", x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)")
ggsave("FIG_ROC_Combined.pdf", autoplot_roc, width = 6.5, height = 5)

# ---- 9) Variable importance (RF and XGBoost) ----
# Fit final models on full data to extract VIP
final_rf_fit <- final_rf %>% fit(dat_model)
final_xgb_fit <- final_xgb %>% fit(dat_model)
final_lasso_fit <- final_lasso %>% fit(dat_model)
final_logit_fit <- final_logit %>% fit(dat_model)
# VIP: Random Forest
p_vip_rf <- vip::vip(final_rf_fit$fit$fit$fit, num_features = 15) +
  labs(title = "Permutation Variable Importance — Random Forest")
ggsave("FIG_Importance_RF.pdf", p_vip_rf, width = 6.5, height = 5)

p_vip_lasso <- vip::vip(final_lasso_fit$fit$fit$fit, num_features = 15) +
  labs(title = "Permutation Variable Importance — LASSO")
p_vip_lasso <- vip::vip(final_logit_fit$fit$fit$fit, num_features = 15) +
  labs(title = "Permutation Variable Importance — Logit")
# VIP: XGBoost
# parsnip stores a xgb.Booster under fit$fit$fit
p_vip_xgb <- vip::vip(final_xgb_fit$fit$fit$fit, num_features = 15) +
  labs(title = "Gain-based Variable Importance — XGBoost")
ggsave("FIG_Importance_XGB.pdf", p_vip_xgb, width = 6.5, height = 5)

# demographic variables to include==========================================================
demo_candidates <- c("age_cont", "age_discret", "gender", "income", "education",
                     "and_mad", "schooling","producing_area")
demo_cols <- intersect(demo_candidates, names(scored_with_clusters))
full_predictors <- c(Fcols, demo_cols)
dat_model <- scored_with_clusters %>%
  select(evoo_choice, all_of(Fcols), all_of(demo_cols)) %>%
  drop_na() %>%
  mutate(evoo_choice = factor(evoo_choice, levels = c(0,1)))

# 2. TRAIN / TEST SPLIT (80/20 stratified)
dat_model<- dat_model %>%
  mutate(evoo_choice = factor(evoo_choice, levels = c(0,1)))
OUTCOME <- "evoo_choice"

spl <- initial_split(dat_model, prop = 0.8, strata = "evoo_choice")

train <- training(spl)
test  <- testing(spl)

rec <- recipe(evoo_choice ~ ., data = train) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

lambda_selected <- 0.00321  

mod <- logistic_reg(
  penalty = lambda_selected,
  mixture = 1,
  mode = "classification"
) %>% 
  set_engine("glmnet")

wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(mod)

# 5. FIT FINAL MODEL ON TRAINING SET
lasso_fit <- fit(wf,data = train)

# 6. PREDICT ON TEST SET
pred <- predict(lasso_fit, test, type = "prob") %>%
  bind_cols(
    predict(lasso_fit, test, type = "class"),
    test %>% select(all_of(OUTCOME))
  ) %>%
  rename(pred_prob = .pred_1,
         pred_class = .pred_class)

# 7. PERFORMANCE METRICS: bias
############################################################
roc_obj <- roc(as.numeric(pred[[OUTCOME]])-1, pred$pred_prob)
# Prediction bias: mean(pred_prob) - mean(actual)
bias <- mean(pred$pred_prob) - mean(as.numeric(pred[[OUTCOME]]) - 1)

# 8. Confusion matrix
cm <- conf_mat(pred, truth = !!sym(OUTCOME), estimate = pred_class)

# 10. Extract coefficients + odds ratios
############################################################
# Extract glmnet model inside workflow
glmnet_fit <- lasso_fit$fit$fit$fit

coef_mat <- coef(glmnet_fit, s = lambda_selected)
coef_df <- tibble(
  term = rownames(coef_mat),
  beta = as.numeric(coef_mat)
) %>% filter(term != "(Intercept)") %>%
  mutate(odds_ratio = exp(beta))

print(coef_df)

############################################################
# Print main metrics
cat("Bias:", bias, "\n")
cm

library(ggplot2)
library(patchwork)

###########################################################
# 1. Training predictions (same structure as test)
###########################################################
pred_train <- predict(lasso_fit, train, type = "prob") %>%
  bind_cols(
    predict(lasso_fit, train, type = "class"),
    train %>% select(evoo_choice)
  ) %>%
  rename(pred_prob = .pred_1,
         pred_class = .pred_class) %>%
  mutate(dataset = "Training")

###########################################################
# 2. Add dataset label to test predictions
###########################################################
pred_test <- pred %>% mutate(dataset = "Test")

###########################################################
# 3. Combine both datasets
###########################################################
pred_all <- bind_rows(pred_train, pred_test)

###########################################################
# 4. Plot: predicted probability vs actual outcome
###########################################################
p <- ggplot(pred_all, aes(x = evoo_choice, y = pred_prob, color = evoo_choice)) +
  geom_jitter(width = 0.1, alpha = 0.25) +
  geom_boxplot(width = 0.4, alpha = 0.3, color = "black") +
  facet_wrap(~ dataset) +
  scale_color_manual(values = c("darkred", "darkgreen")) +
  labs(title = "Predicted Probability vs Actual EVOO Choice",
       x = "Actual EVOO Choice (0 = ROO user, 1 = EVOO user)",
       y = "Predicted Probability of Choosing EVOO") +
  theme_minimal(base_size = 14)

print(p)


#### mapping indicator================================================
library(sf); library(giscoR); library(tidyverse); library(tmap); library(scales)
# Aggregate indicators by NUTS2 NAME_EN or province (use your region labels)
ind <- scored_with_clusters |>
  mutate(region = ifelse(and_mad == 0, "Madrid", "Andalucía")) |>
  group_by(region) |>
  summarise(
    n = n(),
    evoo_mean = mean(evoo_con, na.rm = TRUE),
    EVOO_trust_mean = mean(
      rowMeans(across(starts_with("act_evoo")), na.rm = TRUE),
      na.rm = TRUE
    ),
    EVOO_mean = mean(evoo_uses, na.rm = TRUE),
    taste_pos_share = mean((taste_1 + taste_2 + taste_3)/3, na.rm = TRUE)
  )

# Get Spain NUTS2 polygons (CRS 4326)
nuts2 <- giscoR::gisco_get_nuts(year = 2021, nuts_level = 2, country = "ES") |>
  st_make_valid() |>
  select(NUTS_NAME = NAME_LATN, geometry)

# A tiny crosswalk from your survey label to a NUTS2 name (edit if needed)
cw <- tribble(
  ~region,       ~NUTS_NAME,
  "Andalucía",   "Andalucía",
  "Madrid",      "Comunidad de Madrid"
)
ind_sf <- ind |> 
  left_join(cw, by = "region") |> 
  right_join(nuts2, by = "NUTS_NAME") |> 
  st_as_sf()
# Map 1: EVOO consumption
tmap_mode("plot")
tm_shape(ind_sf) +
  tm_polygons(
    "EVOO_mean",
    title = "uses/weekly",
    palette = "blue",
    style = "quantile",
    legend.reverse = TRUE
  ) +
  tm_borders() +
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE,
    legend.outside.position = "right",
    legend.title.size = 1.0,
    legend.text.size = 0.8
  )

# Map 2: Trust(EVOO)
tm_shape(ind_sf) +
  tm_polygons(
    fill = "EVOO_trust_mean",
    fill.scale = tm_scale_intervals(
      n=2,
      style = "quantile",
      values = "brewer.greens"
    ),
    fill.legend = tm_legend(title = "Mean Trust (EVOO)")
  ) +
  tm_borders() +
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE,
    legend.outside.position = "right"
  )


# Map 3: Taste-positive share
tm_shape(ind_sf) +
  tm_polygons(
    fill = "taste_pos_share",
    fill.scale = tm_scale_intervals(
      style = "quantile",
      values = "brewer.greens"
    ),
    fill.legend = tm_legend(
      title = "Sensitivity to strong EVOO flavour",
      format = tm_format(format = "percent")
    ),
    fill.na = "white",        # <-- missing = white
    fill.na.alpha = 1         # <-- opaque white
  ) +
  tm_borders() +
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE
  )


