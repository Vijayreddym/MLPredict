
prediction <- function()
{
  
  library(Hmisc)
  library(caret)
  library(randomForest)
  library(gbm)
  set.seed(2016)

  
  ##Data Loading From CSV and Cleansing data of NA, DIV/) and empty strings

  trainDataRaw<- read.csv("pml-training.csv", na.strings = c("","NA", "NULL"))
  testDataRaw<- read.csv("pml-testing.csv", na.strings = c("","NA", "NULL"))
  
  ###Basic Data Exploration
  dim(trainDataRaw)
  dim(testDataRaw)

  ## exploration showed they and NAa, Null 
  ## let see the amount NAs and Null present
  naprops <- colSums(is.na(trainDataRaw))/nrow(trainDataRaw)

  #Explorationo NA's
  
  napercent <- colSums(is.na(trainDataRaw))/nrow(trainDataRaw)
  napercent
  
  ### xexcluding data with are not use full for analysis NAs and etc 
#   $ X                       : int  1 2 3 4 5 6 7 8 9 10 ...
#   $ user_name               : Factor w/ 6 levels "adelmo","carlitos",..: 2 2 2 2 2 2 2 2 2 2 ...
#   $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
#   $ raw_timestamp_part_2    : int  788290 808298 820366 120339 196328 304277 368296 440390 484323 484434 ...
#   $ cvtd_timestamp          : Factor w/ 20 levels "02/12/2011 13:32",..: 9 9 9 9 9 9 9 9 9 9 ...
#   $ new_window              : Factor w/ 2 levels "no","yes": 1 1 1 1 1 1 1 1 1 1 ...
#   $ num_window              : int  11 11 11 12 12 12 12 12 12 12 ...
  
  ptrainDataNNA <- trainDataRaw[,colSums(is.na(trainDataRaw)) == 0]

  predicitiontrain <- ptrainDataNNA[,8:length(ptrainDataNNA[1,])]
  dim(predicitiontrain)
  names(predicitiontrain)
  
  nzvColumn <-  which(nearZeroVar(predicitiontrain, saveMetrics = TRUE)$nzv == FALSE)
  predicitiontrainNZV <- predicitiontrain[,nzvColumn]
  
  ## exclude highly correlated variables
  corrMatrix<- cor(predicitiontrainNZV[,sapply(predicitiontrainNZV, is.numeric)])
  highcorrvb<- findCorrelation(corrMatrix, cutoff = .9, verbose = TRUE)
  
  predicitionDataSet<- predicitiontrainNZV[,-highcorrvb]
  dim(predicitionDataSet)
  
  ## split the data for cross validation
  inTrain <- createDataPartition(predicitionDataSet$classe, p = 3/4, list = FALSE)
  training <- predicitionDataSet[inTrain,]
  testing <- predicitionDataSet[-inTrain,]
  
  ## analyze the data with caret package
  modrpart <- train(classe ~., method = "rpart", data = training)
  print(modrpart$finalModel)
  plot(modrpart$finalModel, uniform = TRUE, main = "Classification Tree")
  text(modrpart$finalModel, use.n = TRUE, all = FALSE, cex = 0.8)
  
  ##check accuracy 
  predrpart <- predict(modrpart, testing)
  table(predrpart, testing$classe)
  confusionMatrix(testing$classe, predrpart)$overall['Accuracy']
  ## give pretty bad accuracy
  
  ##gbm ?
  # modgbm<- train(classe ~., method = "gbm", data = training)
  library(gbm)
  modgbm <- gbm(classe ~., data = training, distribution = "multinomial", n.trees = 200, interaction.depth = 4, shrinkage = 0.005)
  predgbm <- predict(modgbm, n.trees = 200, newdata= testing, type = 'response')
  ## The predict returns back the probability for each classe
  ## Below for each row we pick the one with largest probability
  maxpredgbm <- apply(predgbm, 1, which.max)
  
  ## Since 1~5 means A ~ E, we rename them below
  maxpredgbm[which(maxpredgbm == 1)] <- "A"
  maxpredgbm[which(maxpredgbm == 2)] <- "B"
  maxpredgbm[which(maxpredgbm == 3)] <- "C"
  maxpredgbm[which(maxpredgbm == 4)] <- "D"
  maxpredgbm[which(maxpredgbm == 5)] <- "E"
  maxpredgbm <- as.factor(maxpredgbm)
  
  # check the accuracy using confusionMatrix
  confusionMatrix(testing$classe, maxpredgbm)$overall['Accuracy']
  
  library(randomForest)
  modrf <- randomForest(classe~., data = training, ntree=100, importance=TRUE, prox = TRUE)
  
  predrf <- predict(modrf, testing)
  table(predrf, testing$classe)
  confusionMatrix(testing$classe, predrf)$overall['Accuracy']
  
  ##performance <- predict(modrf, testDataRaw)
  ##print(performance)
  
}






