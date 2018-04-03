data1<-read.csv('telecomchurn.csv')
library(ggplot2)
library(corrplot)
View(data1)
str(data1)
data1<-data1[,-4]
data1$Churn<-as.numeric(data1$Churn)
data1$Churn<-ifelse(data1$Churn==1,0,1)
###
churnnum<-sapply(data1,is.numeric)
num<-data1[,churnnum]
num1<-cor(num)
corrplot(num1,method = "circle")
library(caret)
##find columns having correlation gr than 0.9##
dataf=findCorrelation(num1,cutoff = 0.1)
dataf=sort(dataf)
###removing highly correlated columns###
reduced=num[,-c(dataf)]
names<-c(names(reduced),names(data1[,sapply(data1,is.factor)]),"Churn")
data1<-data1[,c(names)]
###Outliers####
lab<-names(reduced)
for (i in 1:ncol(reduced)) {
  png(file = paste("C:\\Users\\MD\\Desktop\\Datasets\\Xg_telecom_churn\\boxplots\\var_", lab[i], ".png", sep=""))
  boxplot(reduced[, i] ,color="blue",main=lab[i])
  dev.off()}
###winsorization###
UB<-quantile(data1$account.length,0.75)+1.5*IQR(data1$account.length)
data1$account.length[data1$account.length>UB]<-UB
UB1<-quantile(data1$total.day.minutes,0.75)+1.5*IQR(data1$total.day.minutes)
LB1<-quantile(data1$total.day.minutes,0.25)-1.5*IQR(data1$total.day.minutes)
data1$total.day.minutes[data1$total.day.minutes>UB1]<-UB1
data1$total.day.minutes[data1$total.day.minutes<LB1]<-LB1
UB2<-quantile(data1$total.eve.minutes,0.75)+1.5*IQR(data1$total.eve.minutes)
LB2<-quantile(data1$total.eve.minutes,0.25)-1.5*IQR(data1$total.eve.minutes)
data1$total.eve.minutes[data1$total.eve.minutes>UB2]<-UB2 
data1$total.eve.minutes[data1$total.eve.minutes<LB2]<-LB2

library(dplyr)#last col first#
data1<-data1[,c(ncol(data1),1:(ncol(data1)-1))]
#Data Visualization#
plot(data1$Churn,data1$international.plan,xlab="Churn",ylab="International calls",main="International calls V/s Churn")
plot(data1$Churn,data1$voice.mail.plan,xlab="Churn",ylab="Voice mail plan",main="Voice mail plan V/s Churn")     
plot(data1$state,data1$Churn,xlab="State",ylab="Churn",main="State vs Churn")
library(caret)
int<-createDataPartition(data1$Churn,p=0.7,list=FALSE)
train<-data1[int,]
test<-data1[-int,]
library(randomForest)
modelrf<-randomForest(Churn~.,data = train,ntree=100)
importance(modelrf)
varImpPlot(modelrf)

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

#########create matrices####
library(Matrix)

trainm<-sparse.model.matrix(Churn~.-1,data=train)
train_label<-train[,"Churn"]
train_matrix<-xgb.DMatrix(data=as.matrix(trainm),label=train_label)
###########
testm<-sparse.model.matrix(Churn~.-1,data=test)
test_label<-test[,"Churn"]
test_matrix<-xgb.DMatrix(data=as.matrix(testm),label=test_label)
#paramters#
nc<-length(unique(train_label))
xgb_params<-list("objective"="multi:softprob","eval_metric"="mlogloss","num_class"=nc)
Watchlist<-list(train=train_matrix,test=test_matrix)
xgbst_model<-xgb.train(params =xgb_params,
                       data=train_matrix,
                       nrounds = 90,
                       watchlist =Watchlist ,
                       eta=0.05)
#training and testing error plot#
err<-data.frame(xgbst_model$evaluation_log)
plot(err$iter,err$train_mlogloss,col='blue',xlab="Number of Iterations",ylab="logloss",main="Train(Logloss Vs No. Iterations)")
lines(err$iter,err$test_mlogloss,col="red")
plot(err$iter,err$test_mlogloss,col='red',xlab="Number of Iterations",ylab="logloss",main="Test(Logloss Vs No. Iterations)")
#Find min testing error and find corresponding iteration#
min(err$test_mlogloss)
err[err$test_mlogloss==0.157603,]
#for eta=0.05,iteration 90 min testing error#
#change nrounds to 90,that is the iteration number for min error#

###Feature imp###
imp<-xgb.importance(colnames(train_matrix),model=xgbst_model)
print(imp)
xgb.plot.importance(imp)
###Prediction and Confusion matrix###
pred<-predict(xgbst_model,newdata = test_matrix)
pred1<-matrix(pred,nrow=nc,ncol=length(pred)/nc)%>%
           t()%>%
           data.frame()%>%
           mutate(label=test_label,max_prob=max.col(.,"last")-1)
                  
table(Prediction=pred1$max_prob,Actual=pred1$label)
library(MLmetrics)
Accuracy(pred1$max_prob,pred1$label)
###testing accuracy of 95.195%



