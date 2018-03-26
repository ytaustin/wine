library(tidyverse)
library(lubridate)
library(caret)

red.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red <- as.tibble(read.csv(red.url, header = TRUE, sep = ";"))
red <- mutate(red, quality = as.factor(red$quality))


ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = fixed.acidity))
ggsave("fixed.acidity.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = volatile.acidity))
ggsave("volatile.acidity.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = citric.acid))
ggsave("citric.acid.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = residual.sugar))
ggsave("residual.sugar.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = chlorides))
ggsave("chlorides.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = free.sulfur.dioxide))
ggsave("free.sulfur.dioxide.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = total.sulfur.dioxide))
ggsave("total.sulfur.dioxide.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = density))
ggsave("density.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = pH))
ggsave("pH.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = sulphates))
ggsave("sulphates.jpg")

ggplot(data = red) +
  geom_boxplot(mapping = aes(x =quality , y = alcohol))
ggsave("alcohol.jpg")

set.seed(121)
trainIndex <-
  createDataPartition(red$quality,
                      p = 0.8,
                      list = FALSE,
                      times = 1)
redTrain <- red[trainIndex, ]
redTest <- red[-trainIndex, ]
scaler <- preProcess(redTrain, method = c("center", "scale"))
redTrain <- predict(scaler, redTrain)
redTest <- predict(scaler, redTest)

knnmodel_fixed.acidity <-
  train(quality ~ fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_fixed.acidity <-
  predict(knnmodel_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_fixed.acidity, redTest$quality)

knnmodel_volatile.acidity <-
  train(quality ~ volatile.acidity, data = redTrain, method = "knn")
redTestPredictions_volatile.acidity <-
  predict(knnmodel_volatile.acidity, redTest)
confusionMatrix(redTestPredictions_volatile.acidity, redTest$quality)

knnmodel_citric.acid <-
  train(quality ~ citric.acid, data = redTrain, method = "knn")
redTestPredictions_citric.acid <-
  predict(knnmodel_citric.acid, redTest)
confusionMatrix(redTestPredictions_citric.acid, redTest$quality)

knnmodel_residual.sugar <-
  train(quality ~ residual.sugar, data = redTrain, method = "knn")
redTestPredictions_residual.sugar <-
  predict(knnmodel_residual.sugar, redTest)
confusionMatrix(redTestPredictions_residual.sugar, redTest$quality)

knnmodel_chlorides <-
  train(quality ~ chlorides, data = redTrain, method = "knn")
redTestPredictions_chlorides <-
  predict(knnmodel_chlorides, redTest)
confusionMatrix(redTestPredictions_chlorides, redTest$quality)

knnmodel_free.sulfur.dioxide <-
  train(quality ~ free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_free.sulfur.dioxide <-
  predict(knnmodel_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_free.sulfur.dioxide, redTest$quality)

knnmodel_total.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide, redTest$quality)

knnmodel_density <-
  train(quality ~ density, data = redTrain, method = "knn")
redTestPredictions_density <-
  predict(knnmodel_density, redTest)
confusionMatrix(redTestPredictions_density, redTest$quality)

knnmodel_pH <-
  train(quality ~ pH, data = redTrain, method = "knn")
redTestPredictions_pH <-
  predict(knnmodel_pH, redTest)
confusionMatrix(redTestPredictions_pH, redTest$quality)

knnmodel_sulphates <-
  train(quality ~ sulphates, data = redTrain, method = "knn")
redTestPredictions_sulphates <-
  predict(knnmodel_sulphates, redTest)
confusionMatrix(redTestPredictions_sulphates, redTest$quality)

knnmodel_alcohol <-
  train(quality ~ alcohol, data = redTrain, method = "knn")
redTestPredictions_alcohol <-
  predict(knnmodel_alcohol, redTest)
confusionMatrix(redTestPredictions_alcohol, redTest$quality)

knnmodel_total.sulfur.dioxide_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_volatile.acidity <-
  train(quality ~ total.sulfur.dioxide+volatile.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_volatile.acidity <-
  predict(knnmodel_total.sulfur.dioxide_volatile.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_volatile.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_citric.acid <-
  train(quality ~ total.sulfur.dioxide+citric.acid, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_citric.acid <-
  predict(knnmodel_total.sulfur.dioxide_citric.acid, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_citric.acid, redTest$quality)

knnmodel_total.sulfur.dioxide_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_residual.sugar, redTest$quality)

knnmodel_total.sulfur.dioxide_chlorides <-
  train(quality ~ total.sulfur.dioxide+chlorides, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_chlorides <-
  predict(knnmodel_total.sulfur.dioxide_chlorides, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_chlorides, redTest$quality)

knnmodel_total.sulfur.dioxide_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_free.sulfur.dioxide, redTest$quality)

knnmodel_total.sulfur.dioxide_density <-
  train(quality ~ total.sulfur.dioxide+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_density <-
  predict(knnmodel_total.sulfur.dioxide_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_density, redTest$quality)

knnmodel_total.sulfur.dioxide_pH <-
  train(quality ~ total.sulfur.dioxide+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_pH <-
  predict(knnmodel_total.sulfur.dioxide_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_pH, redTest$quality)

knnmodel_total.sulfur.dioxide_sulphates <-
  train(quality ~ total.sulfur.dioxide+sulphates, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_sulphates <-
  predict(knnmodel_total.sulfur.dioxide_sulphates, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_sulphates, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol <-
  train(quality ~ total.sulfur.dioxide+alcohol, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol <-
  predict(knnmodel_total.sulfur.dioxide_alcohol, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_volatile.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+volatile.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_volatile.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_volatile.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_volatile.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_citric.acid <-
  train(quality ~ total.sulfur.dioxide+alcohol+citric.acid, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_citric.acid <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_citric.acid, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_citric.acid, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+alcohol+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_residual.sugar, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_chlorides <-
  train(quality ~ total.sulfur.dioxide+alcohol+chlorides, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_chlorides <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_chlorides, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_chlorides, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+alcohol+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_free.sulfur.dioxide, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_density <-
  train(quality ~ total.sulfur.dioxide+alcohol+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_density <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_density, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_pH <-
  train(quality ~ total.sulfur.dioxide+alcohol+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_pH <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_pH, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_volatile.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+volatile.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_volatile.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_volatile.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_volatile.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_citric.acid <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+citric.acid, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_citric.acid <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_citric.acid, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_citric.acid, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_residual.sugar, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_free.sulfur.dioxide, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_density <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_density <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_density, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_pH <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_pH <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_pH, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_citric.acid <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+citric.acid, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_citric.acid <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_citric.acid, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_citric.acid, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_residual.sugar, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_free.sulfur.dioxide, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_density <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_density <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_density, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_pH <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_pH <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_pH, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_residual.sugar, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_free.sulfur.dioxide, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_density <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_density <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_density, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_pH <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_pH <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_pH, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_fixed.acidity <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid+fixed.acidity, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_fixed.acidity <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_fixed.acidity, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_fixed.acidity, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_residual.sugar <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid+residual.sugar, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_residual.sugar <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_residual.sugar, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_residual.sugar, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_free.sulfur.dioxide <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid+free.sulfur.dioxide, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_free.sulfur.dioxide <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_free.sulfur.dioxide, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_free.sulfur.dioxide, redTest$quality)


knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_density <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid+density, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_density <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_density, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_density, redTest$quality)

knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_pH <-
  train(quality ~ total.sulfur.dioxide+alcohol+sulphates+chlorides+volatile.acidity+citric.acid+pH, data = redTrain, method = "knn")
redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_pH <-
  predict(knnmodel_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_pH, redTest)
confusionMatrix(redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid_pH, redTest$quality)


redTest <- mutate(redTest, pred1= redTestPredictions_total.sulfur.dioxide)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred1))
ggsave("round 1.jpg")


redTest <- mutate(redTest, pred2= redTestPredictions_total.sulfur.dioxide_alcohol)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred2))
ggsave("round 2.jpg")


redTest <- mutate(redTest, pred3= redTestPredictions_total.sulfur.dioxide_alcohol_sulphates)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred3))
ggsave("round 3.jpg")


redTest <- mutate(redTest, pred4= redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred4))
ggsave("round 4.jpg")

redTest <- mutate(redTest, pred5= redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred5))
ggsave("round 5.jpg")

redTest <- mutate(redTest, pred6= redTestPredictions_total.sulfur.dioxide_alcohol_sulphates_chlorides_volatile.acidity_citric.acid)
ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred6))
ggsave("round 6.jpg")

ggplot(data = redTest) +
  geom_jitter(mapping = aes(x =quality , y = pred1),color="blue")+
  geom_jitter(mapping = aes(x =quality , y = pred6),color="red")+
  ylab("Round 1 VS round 6")+
  theme(legend.position = "right")
ggsave("round 1vs6.jpg")