
rm(list = ls()) # Clear the workspace to start with a clean environment

# Load necessary libraries for data analysis and forecasting
library("fredr")       # To fetch data from the Federal Reserve Economic Data (FRED)
library("forecast")    # For time series forecasting and analysis
library("lmtest")      # For statistical tests on linear models

# Set your FRED API key for accessing FRED data
fredr_set_key("89ff0fd0319c74a55e2fe79b117d1d82")

# Load electricity price data from FRED (U.S. City Average, per kWh)
data <- fredr(series_id = "APU000072610", observation_start = as.Date("1985-10-01"))

head(data) # Inspection of data

# Set data parameters
s <- 12  # Monthly frequency
start_year <- 1985
start_month <- 10
st <- c(start_year, start_month)
T <- length(data$value)
forecast_length <- s*2

# Convert to time series
data_ts <- ts(data$value, frequency = s, start = st)

# Log transformation to stabilize variance
data_ts_log <- log(data_ts)

data_ts_log <- log(data_ts)
plot(data_ts_log, main = "Log-transformed Electricity Price Data", 
     ylab = "Log(Price)", xlab = "Year")

# Define time index
t <- 1:T          
t2 <- t^2         

# Test for type of trend

# Fit the linear trend model (without quadratic term)
model_linear <- lm(data_ts_log ~ t)

# Fit the quadratic trend model (with quadratic term)
model_quadratic <- lm(data_ts_log ~ t + t2)

# Compare R-squared values
cat("Linear Model R-squared:", summary(model_linear)$r.squared, "\n")
cat("Quadratic Model R-squared:", summary(model_quadratic)$r.squared, "\n")

# Compare residual standard error
cat("Linear Model Residual Standard Error:", summary(model_linear)$sigma, "\n")
cat("Quadratic Model Residual Standard Error:", summary(model_quadratic)$sigma, "\n")


# Isolate residuals in the quadratic model
detrended_data <- residuals(model_quadratic)

# Seasonal adjustment (removing seasonality)
seasonal_decomp <- stl(data_ts_log, s.window = "periodic")

# Create seasonally adjusted data
deseasonalized_data <- seasadj(seasonal_decomp)

# Check for cyclical components with examination of the residuals in trend and seasonally adjusted data
detrended_deseasonalized_data <- residuals(lm(deseasonalized_data ~ t + t2))

# Check for cyclical components by plotting ACF and PACF
acf(detrended_deseasonalized_data, main = "ACF of Detrended and Deseasonalized Data")
pacf(detrended_deseasonalized_data, main = "PACF of Detrended and Deseasonalized Data")

# Fit trend model
model_trend <- model_quadratic

# Seasonal dummy variables
M <- seasonaldummy(data_ts_log) 

# Trend and seasonal matrix
trend_seasonal_matrix <- model.matrix(~ t + t2 + M)

# Fit the ARIMA model with additional lags
model_auto_arima <- auto.arima(data_ts_log, d = 0, seasonal = FALSE, 
                               xreg = trend_seasonal_matrix, stepwise = FALSE, 
                               approximation = FALSE, trace = TRUE,
                               max.p = 10, max.q = 10)

# Display the model summary
summary(model_auto_arima)

checkresiduals(model_auto_arima)

# Forecasting with the fitted model
t_future <- (T + 1):(T + forecast_length)
t2_future <- t_future^2

future_dummies <- seasonaldummy(data_ts_log, forecast_length)

trend_seasonal_forecast <- model.matrix(~ t_future + t2_future + future_dummies)

# Ensure column names match, due to a compile error
colnames(trend_seasonal_forecast) <- colnames(trend_seasonal_matrix)

# Create the forecast
produced_forecast <- forecast(model_auto_arima, h = forecast_length, xreg = trend_seasonal_forecast)

# Inverse the log transformation
produced_forecast$mean <- exp(produced_forecast$mean)
produced_forecast$upper <- exp(produced_forecast$upper)
produced_forecast$lower <- exp(produced_forecast$lower)
produced_forecast$x <- exp(produced_forecast$x)

# Plot the forecast along with original data
model_fit <- exp(data_ts_log - model_auto_arima$residuals)
plot(produced_forecast, main = "Electricity Price Forecast with ARIMA and Seasonal Components", ylab = "Price ($)")
lines(model_fit, col = "green")

# In-sample forecast comparison
data_ts_insample <- subset(data_ts, end = length(data_ts) - forecast_length)
test_subset <- subset(data_ts, start = length(data_ts) - forecast_length + 1)

T_insample <- length(data_ts_insample)
t_insample <- 1:T_insample
t2_insample <- t_insample^2
M_insample <- seasonaldummy(data_ts_insample)

trend_seasonal_forecast_insample <- model.matrix(~ t_insample + t2_insample + M_insample)

# This script does forecast both the best ARIMA model option and the second best.
# Due to only having 5 pages for the assigment the second best model is not included in the final report.

# Fit models to the in-sample matrix
model_trend_ar32 <- Arima(data_ts_insample, order = c(3, 0, 2), include.mean = FALSE, 
                          xreg = trend_seasonal_forecast_insample)

model_trend_ar14 <- Arima(data_ts_insample, order = c(1, 0, 4), include.mean = FALSE, 
                          xreg = trend_seasonal_forecast_insample)

# Forecast for in-sample models
t_future_insample <- (T_insample + 1):(T_insample + forecast_length)
t2_future_insample <- t_future_insample^2
M_future_insample <- seasonaldummy(test_subset)

trend_seasonal_future <- model.matrix(~ t_future_insample + t2_future_insample + M_future_insample)

# Ensure that the column names of the forecast matrix match the training matrix - to solve compile error
colnames(trend_seasonal_future) <- colnames(trend_seasonal_forecast_insample)

# Create in-sample forecasts
forecast_ar32 <- forecast(model_trend_ar32, h = forecast_length, xreg = trend_seasonal_future)
forecast_arma14 <- forecast(model_trend_ar14, h = forecast_length, xreg = trend_seasonal_future)
forecast_arma14

# Plot in-sample forecasts
plot(forecast_ar32, xlim = c(1985, 2028), main = "AR(3,0,2) model with Trend and Seasonality", ylab = "Price in $")
lines(data, col = "red")
plot(forecast_arma14, xlim = c(1985, 2028), main = "ARMA(1,0,4) with Quadratic Trend", ylab = "Price in $", xlab = "Year")
lines(data, col = "blue")

# Test accuracy
accuracy(forecast_ar32, test_subset)
accuracy(forecast_arma14, test_subset)
