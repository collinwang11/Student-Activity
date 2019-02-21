# y = activities

# data preparation --------------------------------------------------------------------------------------
student.por <- read.csv("~/Desktop/student-alcohol-consumption/student-por.csv")
attach(student.por)
str(student.por) 
dim(student.por)
sum(is.na(student.por))
range(student.por$age)

table(student.por$activities)

# model construction  --------------------------------------------------------------------------------------
#set training and testing data 
set.seed(1234)
num_of_data=ceiling(0.1*nrow(student.por)) 
num_of_data
test.index=sample(1:nrow(student.por),num_of_data) 
student.por.testdata=student.por[test.index,] 
student.por.traindata=student.por[-test.index,]


#randomforest test --------------------------------------------------------------------------------------
library(randomForest)
library(caret)
forest = randomForest(activities~., data = student.por.traindata)
forest 
#OOB is 42.47% - with # of trees: 500; no. of variables tried at each split: 5
#Confusion matrix:
#no yes class.error
#no  204 103   0.3355049
#yes 145 132   0.5234657

#forest improved
forest_improve = randomForest(activities~., student.por.traindata, ntree = 800, do.trace = 20, importance = TRUE)
#after improve, the lowest OOB is 40.58%

#predict the testing data to check the OOb error rate
forest_predict = predict(forest_improve, student.por.testdata, type = "class" )
foreset_confusionMatrix = confusionMatrix(forest_predict, student.por.testdata$activities)
foreset_confusionMatrix 
#accuracy is 53.85% (with the improved model)
#P-value: 0.8112

importance(forest_improve)
varImpPlot(forest_improve)
#the important variables plot show us that the related variables are:
#1.	Freetime
#2.	Famerl
#3.	Romantic
#4.	Medu
#5.	G3
#6.	Reason
#7.	Sex
#8.	G2
#9.	Studytime 
#10.	Pstatus
#11.	Gout
#12.Famsup

#decision tree --------------------------------------------------------------------------------------
library(rpart)
library(rpart.plot)
Tree = rpart(activities~., data = student.por.traindata, method = "class")
rpart.plot(Tree)
summary(Tree)
printcp(Tree) #root error: 47.432% 
plotcp(Tree) # only can obtain the 0.93 as the best 

#improve classification tree
tree_full = rpart(activities~., data = student.por.traindata, method = "class", control = rpart.control(cp=0.0000000001))
rpart.plot(tree_full)
summary(tree_full)
tree_full$cptable[which.min(tree_full$cptable[, "xerror"]), "CP"]
min(tree_full$cptable[, "xerror"])

tree_best = rpart(activities~., data = student.por.traindata, control = rpart.control(cp=0.01444043))
print(tree_best)
rpart.plot(tree_best)
summary(tree_best)
printcp(tree_best) #root node error is still 47.432%
plotcp(tree_best)

#check the confusion matrix to see the accuracy and the error rate
library(gmodels)
activities.traindata = student.por$activities[-test.index]
train.predict=factor(predict(tree_best, student.por.traindata, type="class"), levels=levels(activities.traindata))
CrossTable(x = activities.traindata, y = train.predict, prop.chisq=FALSE) 
train.corrcet=sum(train.predict==activities.traindata)/length(train.predict)
train.corrcet # 72.43151% correct rate of training data

activities.testdata=student.por$activities[test.index]
test.predict=factor(predict(tree_best, student.por.testdata, type="class"), levels=levels(activities.testdata))
CrossTable(x = activities.testdata, y = test.predict, prop.chisq=FALSE) 
test.correct=sum(test.predict==activities.testdata)/length(test.predict)
test.correct # 47.69231% correct rate of testing data
#the important variables plot show us that the related variables are "G1", "G2", "G3", "Fedu","Mjob"
#freetime", "romantic", "studytime", "Medu", "sex" and so on.

#logistic regression model --------------------------------------------------------------------------------------
fit_all= glm(activities~., data = student.por.traindata, family = "binomial")
summary(fit_all) 
anova(fit_all, test = "Chisq") # provides analysis of variance table 

#according to the anova test, the siginificant model should be y = shcool, sex, age, pstatus, medu, reason, romantic, freetime, goout, G2

#PCA select variables --------------------------------------------------------------------------------------
#to check the collinearity of the numeric variables
library(ggplot2)
library(ggfortify)
model_vars = student.por[, c("age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel","freetime","goout","Dalc", "Walc", "health", "absences", "G1", "G2","G3")]
pca = prcomp(model_vars, scale = TRUE)
pca
autoplot(pca, data = student.por, loadings = TRUE, col = "grey", loadings.label = TRUE)


#doublecheck with the collinearity 
library(psych)
pairs.panels(model_vars)
library(corrplot)
corrplot(cor(model_vars))

#therefore, we found that goout and freetime is a bit collinearity 
#the model is conducting with "activities = shcool, sex, age, pstatus, medu, reason, romantic, freetime, goout, G2"


fit0= glm(activities~ school+sex + freetime +G2  + romantic +Medu+age+Pstatus+reason+romantic, 
          data = student.por.traindata, family = "binomial")
summary(fit0) # reveals results of preliminary linear regression 

#analysis with model
coefficients(fit0) # model coefficients 
residuals(fit0) # reveals residuals
anova(fit0, test = "Chisq") # provides analysis of variance table 


refitmodel=glm(activities~ sex + freetime +G2  + romantic +Pstatus , family = "binomial", data = student.por.traindata) 
summary(refitmodel)
# fits the regression model based on the ANOVA results: these variables were statistically significant


#predict the testing dataset
fit_pred = predict(refitmodel, data= student.por.testdata, type = "response")
results = as.factor(ifelse(fit_pred>0.5, "yes","no"))
actual = student.por.testdata$activities
fit.test.correct=sum(results==actual)/length(results)
fit.test.correct  #acuracy is 46.58%

#another way to check 
misclassiererror = mean(results != actual)
print(paste("error", misclassiererror)) #error: 53.42%
print(paste("Accuracy", 1 - misclassiererror)) #acuracy is 46.58%

#run the cross validation for model
library(boot)
library(MASS)
cost = function(activities, pi = 0) mean(abs(activities-pi)>0.5)
nodal.glm = glm(activities~sex+freetime+G2+romantic+Pstatus, binomial, data = student.por.testdata)
cv.11.err = cv.glm(student.por.testdata, nodal.glm, cost, K = 11)$delta[1]
cv.11.err[1]
#error rate : 46.15%

accuracy.11.cv = 1 - cv.11.err
accuracy.11.cv
#accuracy rate is 53.857%

coefficients(refitmodel) # model coefficients
confint(refitmodel, level=0.95) # provides confidence intervals for parameters of the model 
fitted(refitmodel) # reveals predictive values
residuals(refitmodel) # reveals residuals
anova(refitmodel) # provides analysis of variance table 
vcov(refitmodel) # provides covariance matrix for parameters of the model 
influence(refitmodel) # runs diagnostics against the model

plot(refitmodel) # plots 4 diagnostic plots for normality, heterosecedasity, influential observations, etc
fits=refitmodel$fitted # stores all fitted values into “fits” variable
resids=refitmodel$residuals #store residuals into “residuals” variable

#activities = shcool, sex, age, pstatus, medu, reason, romantic, freetime, goout, G2"