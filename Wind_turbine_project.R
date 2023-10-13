## ----message=FALSE, warning=FALSE, include=FALSE-------------------------------------------------------------------------------------------------------------------
library(tidyverse)
library(knitr)

# Get the current working directory
current_dir <- getwd()

# Construct the path to the zip file
zip_file_path <- file.path(current_dir, "R80711.zip")

# Unzip the file in the current directory
unzip(zip_file_path, exdir = current_dir)

# Construct the path to the CSV file inside the zip
csv_file_path <- file.path(current_dir, "R80711.csv")

# Read the CSV file
wind_turbine <- read.csv(csv_file_path)

# View the first few rows of the data
head(wind_turbine)



## ---- echo = FALSE-------------------------------------------------------------------------------------------------------------------------------------------------
head(wind_turbine) %>% knitr::kable()


## ----echo=TRUE-----------------------------------------------------------------------------------------------------------------------------------------------------

missing_values <- apply(is.na(wind_turbine), 2, sum)

cat(missing_values)



## ----echo=TRUE-----------------------------------------------------------------------------------------------------------------------------------------------------
cat(colnames(wind_turbine))



## ----echo=FALSE----------------------------------------------------------------------------------------------------------------------------------------------------
library(ggplot2)

# List of variables you want to visualize
variables <- c("temp", "pressure", "humidity", "wind_speed", "wind_deg", "rain_1h", "snow_1h")

# Generate histograms for each variable
lapply(variables, function(var) {
  ggplot(wind_turbine, aes_string(x = var)) + 
    geom_histogram(fill = "skyblue", color = "black", bins = 30) + 
    labs(title = paste("Histogram of", var),
         x = var,
         y = "Frequency") + 
    theme_minimal()
})



## ---- echo = FALSE-------------------------------------------------------------------------------------------------------------------------------------------------

wind_turbine %>% 
  ggplot(aes(x = P_avg)) + 
  geom_histogram(binwidth = 10, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of P_avg", x = "Power (P_avg)", y = "Frequency") +
  theme_minimal()



## ----echo=TRUE-----------------------------------------------------------------------------------------------------------------------------------------------------
library(dplyr)

wind_turbine_clean_1 <- wind_turbine %>% 
  filter(P_avg >= 0)

wind_turbine_clean_1 %>% 
  filter(P_avg > 0) %>% 
  ggplot(aes(x = P_avg)) + 
  geom_histogram(binwidth = 10, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Histogram of P_avg", x = "Power (P_avg)", y = "Frequency") +
  theme_minimal()



## ---- echo = FALSE-------------------------------------------------------------------------------------------------------------------------------------------------

wind_turbine %>% 
  ggplot(aes(x = wind_speed, y = P_avg)) + 
  geom_point(aes(color = wind_speed), alpha = 0.6) + 
  labs(title = "Relationship between Wind Speed and Power Output (P_avg)",
       x = "Wind Speed",
       y = "Power Output (P_avg)") +
  theme_minimal() +
  scale_color_gradient(low = "blue", high = "red")



## ----echo=FALSE----------------------------------------------------------------------------------------------------------------------------------------------------
library(hexbin) 

wind_turbine_clean_1 %>% 
  ggplot(aes(x = wind_speed, y = P_avg)) + 
  geom_hex(aes(fill = after_stat(density)), bins = 30) + 
  scale_fill_viridis_c() + 
  labs(title = "Relationship between Wind Speed and Power Output (P_avg)",
       x = "Wind Speed",
       y = "Power Output (P_avg)") +
  theme_minimal()


## ----echo=FALSE----------------------------------------------------------------------------------------------------------------------------------------------------

# Define a function to generate scatter plots for a given variable
generate_scatter_plot <- function(data, var_name) {
  var_sym <- sym(var_name)
  
  ggplot(data, aes(x = !!var_sym, y = P_avg)) + 
    geom_point(aes(color = !!var_sym), alpha = 0.6) +
    scale_color_gradient(low = "blue", high = "red") + 
    labs(title = paste("Relationship between", var_name, "and Power Output (P_avg)"),
         x = var_name,
         y = "Power Output (P_avg)") +
    theme_minimal()
}

# List of variables to iterate over
variables <- c("temp", "pressure", "humidity", "wind_deg", "rain_1h", "snow_1h")

# Generate and display plots for each variable
plot_list <- lapply(variables, function(var) generate_scatter_plot(wind_turbine_clean_1, var))

# Display each plot in the plot_list
for (p in plot_list) {
  print(p)
}


## ----echo=TRUE-----------------------------------------------------------------------------------------------------------------------------------------------------
library(xgboost)
library(lightgbm)
library(caret)
library(Metrics)

# Sort the data chronologically
wind_turbine_clean_1 <- wind_turbine_clean_1 %>% arrange(Date_time)

# Split the data
train_index <- 1:floor(0.8 * nrow(wind_turbine_clean_1))
train_data <- wind_turbine_clean_1[train_index, ]
test_data <- wind_turbine_clean_1[-train_index, ]

# Define predictor variables and target
predictors <- c("temp", "pressure", "humidity", "wind_speed", "wind_deg", "rain_1h", "snow_1h")
target <- "P_avg"

# Normalize using Min-Max Normalization from the caret package
preProcValues <- preProcess(train_data[, predictors], method = c("range"))
train_data_normalized <- predict(preProcValues, train_data[, predictors])
test_data_normalized <- predict(preProcValues, test_data[, predictors])

# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data_normalized), label = train_data[[target]])
dtest <- xgb.DMatrix(data = as.matrix(test_data_normalized), label = test_data[[target]])

# Hyperparameter tuning for XGBoost
xgb_params_list <- list(
  list(eta = 0.01, max_depth = 4, objective = "reg:linear"),
  list(eta = 0.1, max_depth = 4, objective = "reg:linear"),
  list(eta = 0.3, max_depth = 4, objective = "reg:linear"),
  list(eta = 0.01, max_depth = 6, objective = "reg:linear"),
  list(eta = 0.1, max_depth = 6, objective = "reg:linear"),
  list(eta = 0.3, max_depth = 6, objective = "reg:linear")
)

best_model <- NULL
best_cv_error <- Inf

for (params in xgb_params_list) {
  cv_model <- xgb.cv(params = params, data = dtrain, nfold = 5, nrounds = 100, early_stopping_rounds = 10, verbose = 0)
  min_error <- min(cv_model$evaluation_log$test_rmse_mean)
  
  if (min_error < best_cv_error) {
    best_cv_error <- min_error
    best_model <- cv_model
    best_param <- params
  }
}

# Train the XGBoost model using the best parameters
xgb_model <- xgb.train(params = best_param, data = dtrain, nrounds = 100)
xgb_predictions <- predict(xgb_model, dtest)

# Prepare data for LightGBM
dtrain_lgb <- lgb.Dataset(data = as.matrix(train_data_normalized), label = train_data[[target]])

# Train the LightGBM model:
lgb_model <- lgb.train(data = dtrain_lgb, objective = "regression", nrounds = 100)
lgb_predictions <- predict(lgb_model, data = as.matrix(test_data_normalized))

# Evaluation
evaluate_model <- function(true_values, predictions) {
  residuals <- true_values - predictions
  sse <- sum(residuals^2)
  sst <- sum((true_values - mean(true_values))^2)
  r2 <- 1 - (sse/sst)

  list(
    RMSE = rmse(true_values, predictions),
    R2 = r2,
    MAE = mae(true_values, predictions),
    Normalized_RMSE = rmse(true_values, predictions) / mean(true_values),
    Normalized_MAE = mae(true_values, predictions) / mean(true_values)
  )
}

xgb_evaluation <- evaluate_model(test_data[[target]], xgb_predictions)
lgb_evaluation <- evaluate_model(test_data[[target]], lgb_predictions)

cat("Evaluation for XGBoost:\n")
print(xgb_evaluation)

cat("\nEvaluation for LightGBM:\n")
print(lgb_evaluation)

# After generating the xgb_predictions
test_data$xgb_predictions <- xgb_predictions

# After generating the lgb_predictions
test_data$lgb_predictions <- lgb_predictions



## ----echo=TRUE-----------------------------------------------------------------------------------------------------------------------------------------------------
library(tidyr)

# Convert Date_time to Date format, then group by Date and aggregate P_avg
aggregated_data <- test_data %>%
  mutate(Date = as.Date(Date_time)) %>%
  group_by(Date) %>%
  summarise(P_avg_total = sum(P_avg), 
            xgb_predictions_total = sum(xgb_predictions),
            lgb_predictions_total = sum(lgb_predictions))  # Added for LightGBM predictions

# Convert to long format for ggplot2
aggregated_data_long <- aggregated_data %>%
  gather(key = "Type", value = "Value", P_avg_total, xgb_predictions_total, lgb_predictions_total)

# Plot using ggplot2 with every month labeled on the x-axis
ggplot(aggregated_data_long, aes(x = Date, y = Value, color = Type)) + 
  geom_line() +
  labs(title = "Total Power Output (P_avg), XGBoost and LightGBM Predictions per Day",
       x = "Month",
       y = "Value") +
  theme_minimal() +
  scale_x_date(date_breaks = "1 month", date_labels = "%Y-%m") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_color_manual(values = c("blue", "red", "green"))


