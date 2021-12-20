library(GGally)
library(dplyr)
library(caret)
library(readr)
library(tidyverse)
library(class)
library(partykit)
setwd("C:/Users/anush/OneDrive/Documents/Stat515")
#load and read the data
loan <- read.csv("Bank_Personal_Loan_Modelling.csv")
head(loan)
#check for Null values
colSums(is.na(loan))
#Exploratory analysis 
#Scatter plot for age and income
ggplot(loan,
       aes(x = Age, y = Income)) +
  geom_point(shape=21,size=4,color="black",fill="violet") +
  labs(x="Age",
       y="Incomes",
       title="Age vs Income")
#Scatter plot for experience and income
ggplot(loan,
       aes(x = Experience, y = Income)) +
  geom_point(shape=21,size=4,color="black",fill="violet") +
  labs(x="Experience",
       y="Incomes",
       title="Experience vs Income")
#scatter plot between age and experience
ggplot(loan,
       aes(x = Age, y = Experience)) +
  geom_point(shape=21,size=4,color="black",fill="violet") +
  labs(x="Age",
       y="Experience",
       title="Age vs Experience")
#scatter plot between Income and Mortgage
ggplot(loan,
       aes(x = Income, y = Mortgage)) +
  geom_point(shape=21,size=4,color="black",fill="violet") +
  labs(x="Income",
       y="Mortgage",
       title="Income vs Mortgage")
#scatter plot between Income and CCAvg
ggplot(loan,
       aes(x = Income , y = CCAvg)) +
  geom_point(shape=21,size=4,color="black",fill="violet") +
  labs(x="Income",
       y="CCAvg",
       title="CCAvg vs Income")
#Mortgage distribution
hist (loan$Mortgage ,breaks=10, col="violet", xlab="Mortgage", main=" Mortgage distribution")
#boxplots for education and famiy size
boxplot(Income~Education,
        data=loan,
        main="Different boxplots for each education",
        xlab="Different levels of education",
        ylab=" Income",
        col="violet",
        border="black"
)
boxplot(Income~Family,
        data=loan,
        main="Different boxplots for different family sizes",
        xlab="Different family size",
        ylab=" Income",
        col="violet",
        border="black"
)
#correlation matrix
ggcorr(loan, label = T)

loan_new <- loan %>% 
  mutate(Personal.Loan = as.factor(Personal.Loan), Securities.Account = as.factor(Securities.Account),
         CD.Account = as.factor(CD.Account), Online = as.factor(Online), Family = as.factor(Family),
         CreditCard = as.factor(CreditCard), Education = as.factor(Education)) %>% 
  select(- c(ID, ZIP.Code))

head(loan_new)

#Customers who purchased personal loan
personal_loan <- loan_new %>% 
  filter(Personal.Loan == 1) 
personal_loan
summary(personal_loan)
count(personal_loan)
hist(personal_loan$Income, breaks = 80, xlab = "Income", col = "violet" , main = "Distribution of Income for the customers purchasing loan")

#Adding a new column called performance based on calculating it as Experience/Age

loans_ed <- loan_new %>% 
  mutate(performance = Experience/Age) %>% 
  select(- c(Age, Experience))

loans_ed

#splitting into train test data, cross validation

set.seed(90)

idx <- sample(nrow(loans_ed), nrow(loans_ed)*0.8)

loans_train <- loans_ed[idx,]
loans_test <- loans_ed[-idx,]

logistic.train_label <- loans_ed[idx, "Personal.Loan"]
logistic.test_label <- loans_ed[-idx, "Personal.Loan"]

#checking propertion of train and test data 

prop.table(table(loans_train$Personal.Loan))
prop.table(table(loans_test$Personal.Loan))

#Logistic Regression Model

model_logistic <- glm(formula = Personal.Loan ~ Income + Education + CD.Account + 
                        Family + CreditCard + Online + Securities.Account + CCAvg, 
                      family = "binomial", data = loans_train)

model_logistic %>% 
  summary()

coef(model_logistic)
confint(model_logistic)

#step wise forward logistic regression

loans_none <- glm(formula = Personal.Loan ~ 1, data = loans_train, family = "binomial")
loans_all <- glm(formula = Personal.Loan ~ ., data = loans_train, family = "binomial")
step(object = loans_none, list(lower = loans_none, upper = loans_all), direction = "forward")


#Predicting the customer to apply for personal loan if probability is higher than 0.5

pred <- predict(model_logistic, loans_test, type = "response")
pred_label <-  as.factor(if_else(pred < 0.5, 0, 1))
table(pred_label)

#plot a histogram for the predicted values
hist(predict(model_logistic, loans_test, type = "response"), breaks = 100,
     xlab = "Logistic Model Probability", main = "Histogram for the predicted values on the test data")

#confusion matrix and accuracy of logistic regression model
confusionMatrix(pred_label, loans_test$Personal.Loan, positive = "1")

#KNN
train_label <- loans_train$Personal.Loan
test_label <- loans_test$Personal.Loan
train_knn <- loans_train %>% 
  select_if(is.numeric) %>% 
  scale
test_knn <- loans_test %>% 
  select_if(is.numeric) %>% 
  scale(center = attr(train_knn, "scaled:center"),
        scale = attr(train_knn, "scaled:scale"))
#determining the k value
sqrt(nrow(train_knn))

model_knn <- knn(train = train_knn, test = test_knn, cl = train_label, k = 63)

#confusion matrix and accuracy of KNN
confusionMatrix(model_knn, test_label, positive = "1")

#ROC Plot
library(ROCR)
pred_prob <- predict(object = model_logistic, newdata = loans_test, type = "response")
pred.logistic <- prediction(pred_prob, labels = logistic.test_label)
perf <- performance(prediction.obj = pred.logistic, measure = "tpr", "fpr")
plot(perf)

auc <- performance(pred.logistic ,measure = "auc")
auc@y.values[[1]]


#regression tree
library(rpart)
library(rpart.plot)
install.packages("rpart.plot")
set.seed(1)
train = sample(1:nrow(loan_new), nrow(loan_new)/2)

set.seed(543)
rpart.loan=rpart(Personal.Loan ~.,data = loan_new[train,],
                   method="anova",
                   cp=0.000001)
rpart.loan

#cross-validation-error plot and results
plotcp(rpart.loan)
printcp(rpart.loan)

#using cp=0.033 from the cross validation result to obtain 1-se Loan regression tree
rpart.plot.1se <- prune(rpart.loan,cp=0.033)
rpart.plot(rpart.plot.1se , extra=1,
           roundint=FALSE, digits=3, main="1-se Loan regression tree")

#using cp=0.0001 from the cross validation result to obtain min error tree
rpart.loan.prune <- prune(rpart.loan, cp=0.0001)
rpart.plot(rpart.loan.prune, roundint=FALSE, digits=3, extra=1,
           main="Min-error Loan regression tree")
printcp(rpart.loan.prune)

yhat=predict(rpart.loan.prune, newdata=loan_new[-train,])

loan.test=loan_new[-train,"Personal.Loan"]

