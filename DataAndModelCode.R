#############################################################################################
#                                     Data Manipulation
#############################################################################################

setwd("C:/Users/Andrew/Google Drive/School/Mizzou/Thesis/Data and Code")
library(caret)
library(doParallel)
library(h2o)
library(wordcloud)
library(RColorBrewer)
library(ggplot2)
library(plotly)

cl = makeCluster(detectCores())
registerDoParallel(cl)
set.seed(1)

# Read in Document Term Matrix for Modeling
features = read.table('features.txt', header = T, sep = ',') #717 articles x 465 words
# fix(features)

# # Cats and Dogs Data
# setwd("C:/Users/Andrew")
# cats.dogs = read.table('featuresexample.txt', header = T, sep = ',')
# 
# # Build WordCloud of Features
# new = colSums(features)
# new = as.matrix(new)
# fix(new)
# word_freqs = sort(rowSums(new), decreasing=TRUE)
# dm = data.frame(word=names(word_freqs), freq=word_freqs)
# pal = brewer.pal(8,"Dark2")
# wordcloud(dm$word, dm$freq, min.freq = 1, 
#           random.order=FALSE, colors = pal)

# Read in Volatility Data
vol.data = read.table('Volatility Numbers.txt', header = T) # 717 x 3

# Univariate histograms for Volatility
hist(vol.data$CurrentVol)
hist(vol.data$NextVol)

# Create Categorical Response Variable (Low, Medium, High) for Same Day
vol.data$CurrentCat = ifelse(vol.data$CurrentVol < 4, "Low", 
                             ifelse(vol.data$CurrentVol > 8, "High", "Medium"))
table(vol.data$CurrentCat)

# Create Categorical Response Variable (Low, Medium, High) for Next Day
vol.data$NextCat = ifelse(vol.data$NextVol < 4, "Low", 
                             ifelse(vol.data$NextVol > 8, "High", "Medium"))
table(vol.data$NextCat)

# Final Volatility Data
final.vol = subset(vol.data, select = c(CurrentCat, NextCat)) # 717 x 2
df = as.data.frame(final.vol)

# Data Frame
df = cbind(final.vol, features) # 717 x 467
df = as.data.frame((df))

# Data Frame for Classifying Current
df.curr = subset(df, select = -c(NextCat)) # 717 x 466
df.curr = df.curr[sample(nrow(df.curr)),]

# Data Frame for Classifying Next with Current
df.next = df # 717 x 467
df.next = df.next[sample(nrow(df.next)),]

# Plot over time of Volatility
single.vol = read.table("Single Vol Numbers.txt", header = T)
fix(single.vol)
single.vol$id = row.names(single.vol)

# Scatter Plot
plot(x = single.vol$CurrentVol, y = single.vol$NextVol, 
     xlab = "t - 1", ylab = "t", main = "Plot of Volatility Over Time")

# Time Series Plot
plot(x = single.vol$Date, y = single.vol$NextVol, main = "Volatility Time Series", 
     xlab = "Time", ylab = "Volatility")
lines(single.vol$id, single.vol$NextVol)

# Baseline Model
n = 0
for (i in final.vol$CurrentCat) {
  if (i == final.vol$NextCat) {
    n = n + 1
  }
}
n/717 # Baseline accuracy = 0.4867

# write.table(df.curr, file = "currentdata.csv", sep = ",")
# write.table(df.next, file = "nextdata.csv", sep = ",")

#############################################################################################
#                                     Random Forest
#############################################################################################

## Classifying Current
control1 = trainControl(
              method = "cv", 
              number = 5, 
              allowParallel = T, 
              preProcOptions = list(ncomp = 150))

grid1 = expand.grid(mtry = seq(1, 465, 50))

model1 = train(CurrentCat~., 
              data = df.curr, 
              trControl = control1, 
              method = "ranger", 
              tuneGrid = grid1, 
              importance = "impurity")
model1

# Predict using train after CV
pred1 = predict(model1, df.curr)
out1 = table(pred1, df.curr$CurrentCat)
sum(diag(out1))/nrow(df.curr)
varImp(model1)

### Without PCA
# cv = 5, mtry = 1, accuracy = 0.5089

### With PCA
# cv = 10, mtry = 1, pc = 150, accuracy = 0.7013

# Plot Variable Importance
var1 = c("trump", "election", "could", "say", "go", "you", "million", "get", "include", "state", "obama", "use", "year", "first", "new", "take", "would", "also", "see", "add")
num1 = c(100, 98.34, 95.31, 94.30, 90.16, 88.40, 86.64, 85.86, 84.98, 84.51, 78.34, 77.31, 76.89, 76.81, 76.45, 75.93, 75.50, 74.81, 72.60, 72.42)

var.imp1 = data.frame(cbind(var1, as.numeric(num1)))
fix(var.imp1)
names(var.imp1) = c("variable", "number")
var.imp1$number = as.numeric(var.imp1$number)
write.table(var.imp1, file = "varimp1.csv", sep = ",")

## Classifying Next with Current
control2 = trainControl(
              method = "cv", 
              number = 5, 
              allowParallel = T, 
              preProcOptions = list(ncomp = 150))

grid2 = expand.grid(.mtry = c(0, 1, 2, 3, 4, 5))

model2 = train(NextCat~., 
               data = df, 
               trControl = control2, 
               method = "ranger", 
               tuneGrid = grid2, 
               importance = "impurity")
model2

# Predict using train after CV
pred2 = predict(model2, df)
out2 = table(pred2, df$NextCat)
sum(diag(out2))/nrow(df)
varImp(model2)

# Variable Importance Plot
var2 = c("CurrentCat", "trump", "say", "could", "new", "people", "state", "you", "company", "would", "election", "also", "need", "presidentelect", "face", "go", "two", "told", "high", "security")
num2 = c(100, 62.94, 49.51, 48.42, 46.77, 45.93, 45.81, 45.31, 44.27, 43.54, 43.40, 41.47, 40.54, 40.23, 39.20, 38.42, 37.92, 37.56, 36.78, 36.75)

var.imp2 = data.frame(cbind(var2, as.numeric(num2)))
fix(var.imp2)
names(var.imp2) = c("variable", "number")
write.table(var.imp2, file = "varimp2.csv", sep = ",")


### Without PCA
# cv = 5, mtry = 1, accuracy = 0.5186

### With PCA
# cv = 5, mtry = 1, pc = 150, accuracy = 0.6165

#############################################################################################
#                                     Boosting
#############################################################################################


## Classifying Current
control3 = trainControl(
              method = "cv", 
              number = 5, 
              allowParallel = T, 
              preProcOptions = list(ncomp = 150))

grid3 = expand.grid(.nrounds = seq(1, 10, 2), 
                    .max_depth = seq(1, 400, 100), 
                    .eta = c(0.01, 0.05, 0.1, 0.2), 
                    .gamma = c(0.1, 1, 5, 10), 
                    .colsample_bytree = c(0.1, 1, 5, 10), 
                    .min_child_weight = c(0.1, 1, 5, 10))

model3 = train(CurrentCat~., 
               data = df.curr, 
               trControl = control3, 
               method = "xgbTree", 
               tuneGrid = grid3)
model3

# Predict using train after CV
pred3 = predict(model3, df.curr)
out3 = table(pred3, df.curr$CurrentCat)
sum(diag(out3))/nrow(df.curr)

### Without PCA
# eta = "0.01", max_depth = "101", gamma = "5", colsample_bytree = "1", min_child_weight = "10", num_class = "3"
# accuracy = 0.5495

### With PCA
# eta = "0.2", max_depth = "101", gamma = "5", colsample_bytree = "1", min_child_weight = "0.2", num_class = "3"
# accuracy = 0.5743

## Classifying Next with Current

control4 = trainControl(method = "cv", 
                        number = 5, 
                        allowParallel = T, 
                        preProcOptions = list(thresh = 0.95))

grid4 = expand.grid(.nrounds = seq(1, 10, 2), 
                    .max_depth = seq(1, 400, 100), 
                    .eta = c(0.01, 0.05, 0.1, 0.2), 
                    .gamma = c(0.1, 1, 5, 10), 
                    .colsample_bytree = c(0.1, 1, 5, 10), 
                    .min_child_weight = c(0.1, 1, 5, 10))

model4 = train(NextCat~., 
               data = df, 
               trControl = control4, 
               method = "xgbTree", 
               tuneGrid = grid4)
model4$finalModel

# Predict using train after CV
pred4 = predict(model4, df)
out4 = table(pred4, df$NextCat)
sum(diag(out4))/nrow(df)

### Without PCA
# nrounds = 5, eta = "0.1", max_depth = "1", gamma = "0.1", colsample_bytree = "1", min_child_weight = "0.1", num_class = "3"
# Accuracy = 0.5634

### With PCA
# nrounds = 7, eta = "0.2", max_depth = "1", gamma = "0.1", colsample_bytree = "1", min_child_weight = "10", num_class = "3"
# Accuracy = 0.5648

####################################################################
#                             SVM
####################################################################

### Classifying Current
control5 = trainControl(
  method = "cv", 
  number = 5, 
  allowParallel = T, 
  preProcOptions = list(ncomp = 150))

grid5 = expand.grid(.sigma = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5), 
                    .C = seq(1, 10, 1))

model5 = train(CurrentCat~., 
               data = df.curr, 
               trControl = control5, 
               method = "svmRadial", 
               tuneGrid = grid5)
model5

# Predict using train after CV
pred5 = predict(model5, df.curr)
out5 = table(pred5, df.curr$CurrentCat)
sum(diag(out5))/nrow(df.curr)

### Without PCA
# sigma = 0.01, cost = 5
# Accuracy = 0.7238, nfold = 5

### With PCA
# sigma = 0.001, cost = 3
# Accuracy = 0.7489, nfold = 5, ncomp = 150


### Classifying Next
control6 = trainControl(
  method = "cv", 
  number = 5, 
  allowParallel = T, 
  preProcOptions = list(ncomp = 150))

grid6 = expand.grid(.sigma = c(0.001, 0.005, 0.01, 0.05, 0.1, 0.5), 
                    .C = seq(1, 10, 1))

model6 = train(NextCat~., 
               data = df, 
               trControl = control6, 
               method = "svmRadial", 
               tuneGrid = grid6)
model6

# Predict using train after CV
pred6 = predict(model6, df)
out6 = table(pred6, df$NextCat)
sum(diag(out6))/nrow(df)

### Without PCA
# sigma = 0.001, cost = 1
# Accuracy = 0.6109, nfold = 5

### With PCA
# sigma = 0.001, cost = 3
# Accuracy = 0.7489, nfold = 5, ncomp = 150







##### Without PCA...same results
# Random Forests with only CurrentCat
# accuracy = 0.5523

# Boosting with only CurrentCat
# accuracy = 0.5523

# SVM with only CurrentCat
# accuracy = 0.5523