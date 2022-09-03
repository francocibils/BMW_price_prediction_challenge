################################################################################
######################## BMW Price Prediction Challenge ########################
################################################################################
### Autor: Franco Cibils

# Libraries
library(ggplot2)
library(ggpubr)
library(corrplot)
library(dplyr)
library(forcats)
library(glmnet)
library(rpart)
library(rpart.plot)
library(caret)
library(randomForest)
library(gbm)

# User-defined functions created for this script
my_theme_1 = function(){
  theme_classic() +
    theme(plot.title = element_text(color = 'gray25', size = 18, face = 'bold', hjust = 0.5),
          axis.title.x = element_text(color = 'gray20', size  = 12, face = 'bold'),
          axis.title.y = element_blank(),
          axis.text.x = element_text(face = 'bold'),
          axis.text.y = element_text(face = 'bold'),
          panel.background = element_rect(fill = 'gray95'),
          panel.grid.major = element_line(colour = 'gray90'),
          panel.grid.minor = element_line(colour = 'gray90')) 
}

my_theme_2 = function(){
  theme_classic() +
    theme(plot.title = element_text(color = 'gray25', size = 18, face = 'bold', hjust = 0.5),
          axis.title.x = element_text(color = 'gray20', size  = 12, face = 'bold'),
          axis.title.y = element_text(color = 'gray20', size  = 12, face = 'bold'),
          axis.text.x = element_text(face = 'bold'),
          axis.text.y = element_text(face = 'bold'),
          panel.background = element_rect(fill = 'gray95'),
          panel.grid.major = element_line(colour = 'gray90'),
          panel.grid.minor = element_line(colour = 'gray90')) 
}

my_theme_3 = function(){
  theme(legend.position = c(.95, .95),
        legend.justification = c("right", "top"),
        legend.box.just = "right",
        legend.background = element_rect(color = 'black'),
        legend.box.background = element_rect(fill = 'gray95'),
        legend.title = element_text(face = 'bold', size =14),
        legend.text = element_text(size = 12)) 
}

rmse = function(actual_values, predicted_values){
  ## Computes RMSE
  
  rmse_value = sqrt((sum((actual_values - predicted_values)^2 ) / length(actual_values)))
  
  return(rmse_value)
}

r2 = function(actual_values, predicted_values){
  ## Computes R-squared
  
  # Compute mean of actual values
  mean_actual = mean(actual_values)
  
  # Compute Total Sum of Squares (TSS)
  tss = sum((actual_values - mean_actual)^2)
  
  # Compute Residual Sum of Squares (RSS)
  rss = sum((actual_values - predicted_values)^2)
  
  # Compute R-squared
  r2 = 1 - rss / tss
  
  return(r2)
}

actual_vs_prediction.plot = function(actual_values, prediction_values, model_name = 'Model') {
  
  # Convert prediction and realization values into columns
  dataframe = as.data.frame(cbind(actual_values, prediction_values), make.names = TRUE)
  
  # Figure title
  #title = paste('Actual Price vs', model_name, 'Prediction Price', sep = ' ')
  
  # ggplot figure
  figure = ggplot(dataframe, aes(x = actual_values, y = prediction_values)) + 
    geom_point() + 
    geom_abline(slope = 1, intercept = 0, colour = 'red', size = 1) +
    labs(title = model_name, x = 'Actual price', y = 'Predicted price') +
    my_theme_2()
  
  return(figure)
}

################################################################################
### Loading data
################################################################################
datos = read.csv("C:/Users/franc/Desktop/Universidad/ME/Big Data, Machine Learning and Econometrics/Exam/bmw_pricing_challenge.csv")

################################################################################
### Initial EDA
################################################################################

class(datos)
str(datos)
head(datos)
tail(datos)
summary(datos)
sum(is.na(datos))
names(datos)
dim(datos)

################################################################################
### Initial data preprocessing
################################################################################

# Drop maker_key and model_key features
datos = subset(datos, select = -c(maker_key))

# Changing absurd value
datos[2939, 'mileage'] = 64
datos[4685, 'price'] = 42800
datos[4754, 'price'] = 18500

# Transforming date columns into relevant features
datos$registration_date_year = format(as.Date(datos$registration_date, "%Y-%m-%d"), "%Y")
datos$car_age = 2018 - as.numeric(datos$registration_date_year)

season = as.Date(cut(as.Date(datos$sold_at, "%Y-%m-%d"), "month"))
datos$sold_at_season = factor(quarters(season), 
                              levels = c("Q1", "Q2", "Q3", "Q4"), 
                              labels = c("Winter", "Spring", "Summer", "Fall"))

# Transforming character columns into categorical features
factor_cols = c('fuel', 'paint_color', 'car_type', 'model_key', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8')

for(col in factor_cols){
  datos[[col]] = as.factor(datos[[col]])
}

################################################################################
### Exploring data (EDA)
################################################################################

# Transforming numerical variables into categorical to explore proportions
price_cat = cut(datos$price, breaks = c(0, 10000, 20000, 30000, 200000), labels = c('<10K', '10K-20K', '20K-30K', '>30K'))
mileage_cat = cut(datos$mileage, breaks = c(0, 10000, 50000, 100000, 150000, 1000000), labels = c('<10K', '10K-50K', '50K-100K', '100-150K', '>150K'))
age_cat = cut(datos$car_age, breaks = c(0, 2, 5, 10, 30), labels = c('<2 years', '2-5 years', '5-10 years', '>10 years'))

# Exploring categorical variables
table_fuel = table(datos$fuel)
table_car_type = table(datos$car_type)
table_price = table(price_cat)
table_mileage = table(mileage_cat)
table_age = table(age_cat)
table_color = table(datos$paint_color)

prop.table(table_price) # Replace with each table

# Correlation between numeric variables
numeric_columns = c('mileage', 'car_age', 'engine_power', 'price')
correlations = cor(datos[,numeric_columns])
res.95 = cor.mtest(datos[,numeric_columns], conf.level = 0.95)

corrplot(correlations, 
         method = 'color', 
         type = 'upper', 
         order = 'hclust', 
         addCoef.col = 'black',
         p.mat = res.95$p,
         diag = FALSE)

# Univariate variables' figures
ggplot(data = datos, aes(price)) + 
  geom_histogram(bins = 50, fill = 'lightslategray') +
  labs(title = 'Car price histogram', x = 'Price') +
  my_theme_1() 

ggplot(data = datos, aes(price)) + 
  geom_histogram(bins = 70, fill = 'lightslategray') + 
  labs(title = 'Car Price Histogram - Price range: 0-50k', x = 'Price') + 
  coord_cartesian(xlim = c(0,50000)) +
  my_theme_1()

ggplot(data = datos, aes(mileage)) + 
  geom_histogram(bins = 50, fill = 'lightslategray') + 
  labs(title = 'Mileage Histogram', x = 'Mileage') + 
  my_theme_1()

ggplot(data = datos, aes(mileage)) + 
  geom_histogram(bins = 100, fill = 'lightslategray') + 
  coord_cartesian(xlim = c(0,500000)) + 
  labs(title = 'Mileage Histogram - Mileage range: 0-50k', x = 'Mileage') + 
  my_theme_1()

ggplot(data = datos, aes(engine_power)) + 
  geom_histogram(bins = 25, fill = 'lightslategray') + 
  labs(title = 'Engine power histogram', x = 'Engine power') +
  my_theme_1()

ggplot(data = datos, aes(car_age)) + 
  geom_histogram(bins = 28, fill = 'lightslategray') + 
  labs(title = 'Car age histogram', x = 'Years old') +
  my_theme_1() + 
  scale_x_continuous(breaks = seq(0, 25, 5))

ggplot(data = datos, aes(fuel, fill = fuel)) + 
  geom_bar() +
  labs(title = 'Fuel barchart', x = 'Fuel type') + 
  my_theme_1() +
  scale_x_discrete(labels = c('Diesel', 'Electro', 'Hybrid petrol', 'Petrol')) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(car_type, fill = car_type)) + 
  geom_bar() + 
  labs(title = 'Car type barchart', x = 'Car type') + 
  my_theme_1() + 
  scale_x_discrete(labels = c('Convertible', 'Coupe', 'Estate', 'Hatchback', 'Sedan', 'Subcompact', 'SUV', 'Van')) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(sold_at_season, fill = sold_at_season)) + 
  geom_bar() + 
  labs(title = 'Sold at season', x = 'Season') + 
  my_theme_1() +
  theme(legend.position = 'none')

#feature_columns = c()
#feature_columns_bar = c()
#for(i in 1:8){
#  feature_name = paste('feature', i, 'bar', sep = '_')
#  feature_columns_bar[i] = feature_name
#  feature_column = paste('feature', i, sep = '_')
#  feature_columns[i] = feature_column
#  assign(feature_columns_bar[i], ggplot(data = datos, aes(unlist(datos[feature_columns[i]]), fill = unlist(datos[feature_columns[i]]))) +  geom_bar() + labs(title = paste('Feature', i, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none'))
#}

feature_1_bar = ggplot(data = datos, aes(x = feature_1, fill = feature_1)) +  geom_bar() + labs(title = paste('Feature', 1, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_2_bar = ggplot(data = datos, aes(x = feature_2, fill = feature_2)) +  geom_bar() + labs(title = paste('Feature', 2, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_3_bar = ggplot(data = datos, aes(x = feature_3, fill = feature_3)) +  geom_bar() + labs(title = paste('Feature', 3, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_4_bar = ggplot(data = datos, aes(x = feature_4, fill = feature_4)) +  geom_bar() + labs(title = paste('Feature', 4, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_5_bar = ggplot(data = datos, aes(x = feature_5, fill = feature_5)) +  geom_bar() + labs(title = paste('Feature', 5, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_6_bar = ggplot(data = datos, aes(x = feature_6, fill = feature_6)) +  geom_bar() + labs(title = paste('Feature', 6, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_7_bar = ggplot(data = datos, aes(x = feature_7, fill = feature_7)) +  geom_bar() + labs(title = paste('Feature', 7, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')
feature_8_bar = ggplot(data = datos, aes(x = feature_8, fill = feature_8)) +  geom_bar() + labs(title = paste('Feature', 8, sep = ' ')) + my_theme_1() + theme(axis.title.x = element_blank(), legend.position = 'none')

ggarrange(feature_1_bar, feature_2_bar, feature_3_bar, feature_4_bar, feature_5_bar, feature_6_bar, feature_7_bar, feature_8_bar, ncol = 3, nrow = 3)

# More than two variables figures
ggplot(data = datos, aes(fuel, price, fill = fuel)) + 
  geom_boxplot() + 
  my_theme_2() +
  labs(title = 'Fuel/Price boxplot', x = 'Fuel type', y = 'Price') +
  scale_y_continuous(breaks = seq(0, 100000, 10000))  +
  scale_x_discrete(labels = c('Diesel', 'Electro', 'Hybrid petrol', 'Petrol')) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(car_type, price, fill = car_type)) + 
  geom_boxplot() +
  my_theme_2() +
  labs(title = 'Car type/Price boxplot', x = 'Car type', y = 'Price') +
  scale_y_continuous(breaks = seq(0, 100000, 10000))  +
  scale_x_discrete(labels = c('Convertible', 'Coupe', 'Estate', 'Hatchback', 'Sedan', 'Subcompact', 'SUV', 'Van')) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(sold_at_season, price, fill = sold_at_season)) + 
  geom_boxplot() +
  my_theme_2() +
  labs(title = 'Sold at season/Price boxplot', x = 'Season', y = 'Price') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(paint_color, price, fill = paint_color)) + 
  geom_boxplot() + 
  my_theme_2() +
  labs(title = 'Paint color/Price boxplot', x = 'Color', y = 'Price') + 
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_x_discrete(labels = c('Beige', 'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Orange', 'Red', 'Silver', 'White')) +
  scale_fill_manual(values = c('tan1', 'black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'lightsteelblue3', 'white')) +
  theme(legend.position = 'none')

ggplot(data = datos, aes(mileage, price, color = fuel)) + 
  geom_point(alpha = 0.7) + 
  coord_cartesian((xlim = c(0,450000)), ylim = c(0,100000)) +
  labs(title = 'Price/Mileage scatterplot by Fuel type', x = 'Mileage', y = 'Price', color = 'Fuel type') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_color_discrete(name = 'Fuel type', labels = c('Diesel', 'Electro', 'Hybrid petrol', 'Petrol')) +
  my_theme_2() +
  my_theme_3()

ggplot(data = datos, aes(mileage, price, color = car_type)) + 
  geom_point(alpha = 0.7) + 
  coord_cartesian((xlim = c(0,450000)), ylim = c(0,100000)) +
  labs(title = 'Price/Mileage scatterplot by Car type', x = 'Mileage', y = 'Price', color = 'Car type') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_color_discrete(name = 'Car type', labels = c('Convertible', 'Coupe', 'Estate', 'Hatchback', 'Sedan', 'Subcompact', 'SUV', 'Van')) +
  my_theme_2() +
  my_theme_3()

ggplot(data = datos, aes(mileage, price, color = car_age)) + 
  geom_point(alpha = 0.8) + 
  coord_cartesian((xlim = c(0,450000)), ylim = c(0,100000)) +
  labs(title = 'Price/Mileage scatterplot by age', x = 'Mileage', y = 'Price', color = 'Car age') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_color_gradient(low = 'lightblue', high = 'black') +
  my_theme_2() +
  my_theme_3() 

ggplot(data = datos, aes(car_age, price, color = car_type)) + 
  geom_point(position = 'jitter', alpha = 0.5) +
  labs(title = 'Price/Age scatterplot by car type', x = 'Age', y = 'Price', color = 'Car type') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_x_continuous(breaks = seq(0, 30, 2)) +
  scale_color_discrete(name = 'Car type', labels = c('Convertible', 'Coupe', 'Estate', 'Hatchback', 'Sedan', 'Subcompact', 'SUV', 'Van')) +
  my_theme_2() +
  my_theme_3()

ggplot(data = datos, aes(engine_power, price, color = car_type)) + 
  geom_point(position = 'jitter', alpha = 0.9) +
  labs(title = 'Price/Engine power by car type', x = 'Engine power', y = 'Price', color = 'Car type') +
  scale_y_continuous(breaks = seq(0, 100000, 10000)) +
  scale_x_continuous(breaks = seq(0, 400, 50)) +
  scale_color_discrete(name = 'Car type', labels = c('Convertible', 'Coupe', 'Estate', 'Hatchback', 'Sedan', 'Subcompact', 'SUV', 'Van')) +
  my_theme_2() +
  my_theme_3()

################################################################################
### Further data pre-processing
################################################################################

# Collapsing categorical features into fewer categories: fuel, paint_color and car_type
fuel_cat_low = c('diesel', 'petrol'); fuel_cat_high = c('electro', 'hybrid_petrol')
color_cat_normal = c('beige', 'black', 'blue','brown', 'grey', 'red', 'silver', 'white'); color_cat_high = c('orange'); color_cat_low = c('green')
car_cat_high = c('coupe', 'suv', 'van'); car_cat_low = c('convertible', 'estate', 'hatchback', 'sedan', 'subcompact')

datos$fuel_cat = fct_collapse(datos$fuel, high = fuel_cat_high, low = fuel_cat_low)
datos$color_cat = fct_collapse(datos$paint_color, normal = color_cat_normal, high = color_cat_high, low = color_cat_low)
datos$car_cat = fct_collapse(datos$car_type, high = car_cat_high, low = car_cat_low)

################################################################################
### Price prediction
################################################################################

set.seed(1)
datos_nrow = nrow(datos)
columns_to_use_simple = c('mileage', 'engine_power', 'car_age', 'fuel_cat', 'car_cat', 'color_cat', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'price')
columns_to_use_complex = c('model_key', 'mileage', 'engine_power', 'car_age', 'fuel_cat', 'car_cat', 'color_cat', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'price')

datos_use_simple = datos[, columns_to_use_simple]
datos_use_complex = datos[, columns_to_use_complex]

id = sample(datos_nrow, round(0.7 * datos_nrow, 0), replace = F)

# Train-test split for simple models
X = model.matrix( ~ .-1, datos_use_simple);
y = as.matrix(datos_use_simple[ ,15])

x_train = X[id, c(1:4,6:16)]
x_test = X[-id, c(1:4,6:16)]
y_train = y[id]
y_test = y[-id]

# Train-test for complex models
train_complex = datos_use_complex[id,]
test_complex = datos_use_complex[-id,]

## Linear Regression
reg.lin = lm(y_train ~ ., data = as.data.frame(cbind(x_train, y_train), make.names = T))
summary(reg.lin)

pred.out.reg.lin = predict(reg.lin, newdata = as.data.frame(cbind(x_test, y_test), make.names = T))
rmse_lm.out.sample = rmse(y_test, pred.out.reg.lin)
r2_lm.out.sample = r2(y_test, pred.out.reg.lin)

## Ridge 
seq_lambda = (seq(150 , 0 , length =500))^2 

ridge = glmnet(x_train, y_train, alpha = 0, family = 'gaussian')
plot(ridge, xvar = 'lambda')
plot(ridge, xvar = 'norm')

ridge.model = cv.glmnet(x_train, y_train,
                   family = 'gaussian', 
                   lambda = seq_lambda,          
                   type.measure = c('mse'),  
                   alpha = 0,                 
                   nfolds = 10)               
ridge.model
plot (ridge.model)
bestlam_ridge = ridge.model$lambda.min

ridge.best_model = glmnet(x = x_train , 
                     y = y_train,
                     family = 'gaussian', 
                     alpha = 0 ,           
                     lambda = bestlam_ridge, #151.897
                     standardize=TRUE)

pred.out.ridge = predict(ridge.best_model, s = bestlam_ridge, newx = x_test, type = 'response')
rmse_ridge.out.sample = rmse(y_test, pred.out.ridge)
r2_ridge.out.sample = r2(y_test, pred.out.ridge)

## Lasso
lasso = glmnet(x_train, y_train, family = 'gaussian', alpha = 1)
plot(lasso, xvar = 'lambda')
plot(lasso, xvar = 'norm')

lasso.model = cv.glmnet(x_train, y_train,
                   family = 'gaussian',       
                   lambda = seq_lambda,       
                   type.measure = c('mse'),   
                   alpha = 1,                 
                   nfolds = 10)
lasso.model
plot(lasso.model)
bestlam_lasso = lasso.model$lambda.min

lasso.best_model = glmnet(x = x_train , 
                          y = y_train,
                          family = 'gaussian', 
                          alpha = 1,           
                          lambda = bestlam_lasso, #20.33124
                          standardize=TRUE)

pred.out.lasso = predict(lasso.best_model, s = bestlam_lasso, newx = x_test, type = 'response')
rmse_lasso.out.sample = rmse(y_test, pred.out.lasso)
r2_lasso.out.sample = r2(y_test, pred.out.lasso)

## Elastic net
seq_alpha = seq(0.05, 0.95, by = 0.05)
rmse_elastic.val = c()
bestlam_elastic = c()

for(i in 1:length(seq_alpha)){
  elasticnet.model = cv.glmnet(x_train, y_train,
                           family = 'gaussian', 
                           lambda = seq_lambda,
                           type.measure = c('mse'),
                           alpha = seq_alpha[i],
                           nfolds = 10)         
  
  bestlam_elastic[i] = elasticnet.model$lambda.min
  
  elasticnet.best_model = glmnet(x = x_train , 
                            y = y_train,
                            family = 'gaussian', 
                            alpha = seq_alpha[i],           
                            lambda = bestlam_elastic[i],
                            standardize=TRUE)
  
  pred.out.elasticnet = predict(elasticnet.best_model, s = bestlam_elastic[i], newx = x_train, type = 'response')
  rmse_elastic.val[i] = sqrt((sum((y_train - pred.out.elasticnet)^2 ) / length(y_test)))
  print(i)
}

rmse_elastic.in.sample = rmse_elastic.val[which(rmse_elastic.val == min(rmse_elastic.val))]
best_alpha = seq_alpha[which(rmse_elastic.val == min(rmse_elastic.val))]
best_lambda = bestlam_elastic[which(rmse_elastic.val == min(rmse_elastic.val))]

elasticnet.best_model = glmnet(x = x_train , 
                               y = y_train,
                               family = 'gaussian', 
                               alpha = best_alpha, #0.95         
                               lambda = best_lambda, #9.036108
                               standardize=TRUE)

pred.out.elasticnet = predict(elasticnet.best_model, s = best_lambda, newx = x_test, type = 'response')
rmse_elastic.out.sample = rmse(y_test, pred.out.elasticnet)
r2_elastic.out.sample = r2(y_test, pred.out.elasticnet)

## Decision Tree
dt_model = rpart(price ~ ., data = train_complex,
                 method = 'anova',
                 control = rpart.control(cp = 0.005, minsplit= 10, xval = 10, max_depth = 50))
summary(dt_model)
x11()
plot(dt_model,uniform=T,branch=0.1,compress=T,margin=0.1,cex=.5)
text(dt_model,all=T,use.n=T,cex=.7)

rpart.plot(dt_model, type = 2, cex = 0.5)

dt_model$variable.importance / sum(dt_model$variable.importance)
plotcp(dt_model)
printcp(dt_model)

pred.out.dt = predict(dt_model, newdata = test_complex, type = 'vector')
rmse_dt.out.sample = rmse(y_test, pred.out.dt)
r2_dt.out.sample = r2(y_test, pred.out.dt)

## Bagging
values.nbagg = c(100, 200, 300, 400, 500, 1000, 3000) # Best: 100
values.minsplit = c(5, 10, 15, 20, 50, 100, 200, 500) # Best: 5
grid_bagging = expand.grid(values.nbagg = values.nbagg, values.minsplit = values.minsplit)
bagging.mse.oob = c()

for(i in 1:dim(grid_bagging)[1]){
  bagging.model = train(price ~ .,
                        data = train_complex,
                        method = 'treebag',
                        trControl = trainControl(method = "oob"), keepX = T,
                        nbagg = grid_bagging[i, 1],
                        control = rpart.control(minsplit = grid_bagging[i,2], cp = 0.001))
  
  bagging.mse.oob[i] =  unlist(bagging.model$results[1])
  print(i)
}

cbind(grid_bagging, bagging.mse.oob)
min_ecm_bagging = which(bagging.mse.oob == min(bagging.mse.oob))

bagging.best_model = train(price ~ .,
                           data = train_complex,
                           method = 'treebag',
                           trControl = trainControl(method = 'oob'), keepX = T,
                           nbagg = grid_bagging[min_ecm_bagging, 1], #100
                           control = rpart.control(minsplit = grid_bagging[min_ecm_bagging, 2], cp = 0.001)) #5

bagging_varimp = varImp(bagging.best_model) 
dotPlot(bagging_varimp)

pred.out.bagging = predict(bagging.best_model, newdata = test_complex)
rmse_bagging.out.sample = rmse(y_test, pred.out.bagging)
r2_bagging.out.sample = r2(y_test, pred.out.bagging)

## Random Forest
values.mtry = c(3, 4, 5, 6, 7, 8, 9)
values.maxnode = c(5, 10, 20, 50, 100, 200)
values.ntree = c(100, 200, 500, 1000, 3000)
values.nodesize = c(5, 10, 20, 50, 100, 500)
grid_rf = expand.grid(values.mtry = values.mtry, values.maxnode = values.maxnode, values.ntree = values.ntree, values.nodesize = values.nodesize) 
rf.mse.in_sample = c()

for(i in 1:dim(grid_rf)[1]){
  rf.model  = randomForest(y_train ~ .,
                             data = as.data.frame(cbind(x_train, y_train), make.names = T),
                             mtry = grid_rf[i,1],
                             ntree = grid_rf[i,3],
                             sample = 0.5*nrow(x_train), 
                             maxnodes = grid_rf[i,2], 
                             nodesize = grid_rf[i,4], 
                             importance=F, 
                             proximity =F)  
  
  rf.mse.in_sample[i] = mean((y_train - rf.model$predicted)^2) 
  print(i)
}

cbind(grid_rf,rf.mse.in_sample)
min_ecm_rf = which(rf.mse.in_sample == min(rf.mse.in_sample))

rf.best_model = randomForest(y_train ~ .,
                            data = as.data.frame(cbind(x_train, y_train), make.names = T),
                            mtry = grid_rf[min_ecm_rf, 1], #9
                            ntree = grid_rf[min_ecm_rf, 3], #500
                            sample = 0.5*nrow(y_train), 
                            maxnodes = grid_rf[min_ecm_rf, 2], #200
                            nodesize = grid_rf[min_ecm_rf, 4], #20
                            importance = F, 
                            proximity = F)

pred.out.rf = predict(rf.best_model, newdata =  as.data.frame(cbind(x_test, y_test), make.names = T))
rmse_rf.out.sample = rmse(y_test, pred.out.rf)
r2_rf.out.sample = r2(y_test, pred.out.rf)

## Gradient Boosting Machine
values.shrinkage = c(0.3, 0.1, 0.05, 0.01, 0.005, 0.001) # Best: 0.05
values.interaction_depth = c(1, 3, 5, 7, 10, 15, 20, 30) # Best: 15
values.minobsinnode = c(5, 10, 20, 50, 100) # Best: 5
grid_gbm = expand.grid(values.shrinkage = values.shrinkage, 
                       values.interaction_depth = values.interaction_depth, 
                       values.minobsinnode = values.minobsinnode, 
                       optimal_trees = NA, 
                       min_DEV = NA)

for(i in 1:dim(grid_gbm)[1]){
  gbm.model = gbm(price ~ .,
                data = train_complex, 
                distribution = "gaussian", 
                n.trees = 5000,            
                shrinkage = grid_gbm[i, 1], 
                interaction.depth = grid_gbm[i, 2],
                n.minobsinnode = grid_gbm[i, 3],
                train.fraction = 0.8,      # delta
                bag.fraction = 0.7,        # eta 
                cv.folds = 10, 
                verbose = T)
  
  grid_gbm$optimal_trees[i] = which.min(gbm.model$valid.error)
  grid_gbm$min_DEV[i] = min(gbm.model$valid.error)
  print(i)
}

min_ecm_gbm = which(grid_gbm$min_DEV == min(grid_gbm$min_DEV))

gbm.best_model = gbm(price ~ .,
                data = train_complex, 
                distribution = "gaussian",
                n.trees = 5000, 
                shrinkage = grid_gbm[min_ecm_gbm, 1], #0.05
                interaction.depth = grid_gbm[min_ecm_gbm, 2], #15
                n.minobsinnode = grid_gbm[min_ecm_gbm, 3], #5
                train.fraction = 0.8, 
                bag.fraction = 0.7, 
                cv.folds = 10, 
                verbose = T)

summary(gbm.best_model)
which.min(gbm.best_model$valid.error)
min(gbm.best_model$valid.error)

best.iter = gbm.perf(gbm.best_model, method = 'cv')
title(main='GBM - MSE Estimations')
summary(gbm.best_model, n.trees  = best.iter)

pred.out.gbm = predict(gbm.best_model, n.trees = best.iter, newdata = test_complex)
rmse_gbm.out.sample = rmse(y_test, pred.out.gbm)
r2_gbm.out.sample = r2(y_test, pred.out.gbm)

## Out-of-sample performance comparison (RMSE and R-Squared measures for each model)

models = c('Linear Regression',
           'Ridge',
           'Lasso', 
           'Elastic Net', 
           'Decision Tree', 
           'Bagging',
           'Random Forest',
           'GBM')

models_rmse.values = round(c(rmse_lm.out.sample, 
                       rmse_ridge.out.sample, 
                       rmse_lasso.out.sample, 
                       rmse_elastic.out.sample, 
                       rmse_dt.out.sample,
                       rmse_bagging.out.sample,
                       rmse_rf.out.sample,
                       rmse_gbm.out.sample), 2)

models_r2.values = round(c(r2_lm.out.sample, 
                     r2_ridge.out.sample, 
                     r2_lasso.out.sample, 
                     r2_elastic.out.sample, 
                     r2_dt.out.sample,
                     r2_bagging.out.sample,
                     r2_rf.out.sample,
                     r2_gbm.out.sample), 4)

models_performance = as.data.frame(cbind(models, models_rmse.values, models_r2.values))
colnames(models_performance) = c('Model', 'RMSE', 'R-Squared')
models_performance

ggplot(models_performance, aes(x = reorder(models, -models_rmse.values), y = models_rmse.values, fill = models)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label = models_rmse.values), colour = 'white', hjust = 1.1, fontface = 'bold') +
  labs(title = 'Models performance comparison - RMSE', x = 'Models', y = 'RMSE') +
  coord_flip() +
  my_theme_2() + 
  theme(legend.position = 'none')

ggplot(models_performance, aes(x = reorder(models, models_r2.values), y = models_r2.values, fill = models)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label = models_r2.values), colour = 'white', hjust = 1.1, fontface = 'bold') +
  labs(title = 'Models performance comparison - R-Squared', x = 'Models', y = 'R-Squared') +
  coord_flip() +
  my_theme_2() +
  theme(legend.position = 'none')

## Actual Price vs Model Prediction Price - Scatter plot figures
models_names = c('lr', 'ridge', 'lasso', 'elastic', 'dt', 'bagging', 'rf', 'gbm')
models_predictions = cbind(pred.out.reg.lin, pred.out.ridge = pred.out.ridge, pred.out.lasso = pred.out.lasso, pred.out.elasticnet = pred.out.elasticnet, pred.out.dt, pred.out.bagging, pred.out.rf, pred.out.gbm)
colnames(models_predictions) = models
for(i in 1:length(models)){
  model_name = paste('act_vs_pred.', models_names[i], sep = '')
  assign(model_name, actual_vs_prediction.plot(y_test, models_predictions[,i], models[i]))
}

act_vs_pred = ggarrange(act_vs_pred.lr, act_vs_pred.ridge, act_vs_pred.lasso, act_vs_pred.elastic, act_vs_pred.dt, act_vs_pred.bagging, act_vs_pred.rf, act_vs_pred.gbm, ncol = 3, nrow = 3)
annotate_figure(act_vs_pred, top = text_grob('Actual vs Predicted Price for each model',color = 'dodgerblue4', face = 'bold', size = 18))

################################################################################
#################################### FIN #######################################
################################################################################