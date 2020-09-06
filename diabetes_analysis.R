attachp

library(glmnet)
library(MLmetrics)
library(missForest)
library(class)
library(dplyr)
library(plyr)
library(data.table)
library(caret)
library(xgboost)
library(gbm)
library(BBmisc)
library(MASS)
library(DMwR)
library(corrplot)

diabetes = read.csv("../input/diabetes/diabetic_data.csv", na.strings = c("?")) ##replace ? with NA
diabetes <- read.csv("C:/Users/semed/Music/dataset_diabetes/diabetic_data.csv", na.strings = c("?"),header=FALSE)
View(diabetes)
summary(diabetes)
# Data Cleaning and Wrangling
## Removing duplicate patients encounter and only take the first observation to avoid bias
diabetes <- diabetes[!duplicated(diabetes$patient_nbr),]

diabetes$visits = diabetes$number_outpatient + diabetes$number_emergency + diabetes$number_inpatient
readmitted = diabetes$readmitted
diabetes <- subset(diabetes, select =-c(readmitted))
diabetes$readmitted = readmitted
head(diabetes)

diabetes$age <- ifelse(diabetes$age == "[0-10)",  0, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[10-20)", 10, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[20-30)", 20, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[30-40)", 30, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[40-50)", 40, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[50-60)", 50, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[60-70)", 60, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[70-80)", 70, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[80-90)", 80, diabetes$age);
diabetes$age <- ifelse(diabetes$age == "[90-100)", 90, diabetes$age);

diabetes$max_glu_serum <- ifelse(diabetes$max_glu_serum == "None",  0, diabetes$max_glu_serum);
diabetes$max_glu_serum <- ifelse(diabetes$max_glu_serum == "Norm",  100, diabetes$max_glu_serum);
diabetes$max_glu_serum <- ifelse(diabetes$max_glu_serum == ">200",  200, diabetes$max_glu_serum);
diabetes$max_glu_serum <- ifelse(diabetes$max_glu_serum == ">300",  300, diabetes$max_glu_serum);

diabetes$A1Cresult <- ifelse(diabetes$A1Cresult == "None",  0, diabetes$A1Cresult);
diabetes$A1Cresult <- ifelse(diabetes$A1Cresult == "Norm",  5, diabetes$A1Cresult);
diabetes$A1Cresult <- ifelse(diabetes$A1Cresult == ">7",    7, diabetes$A1Cresult);
diabetes$A1Cresult <- ifelse(diabetes$A1Cresult == ">8",    8, diabetes$A1Cresult);

## Remove unique identifiers "encounter_id", "patient_nbr"
diabetes <- diabetes[,-c(1,2)]

## Remove patients coded as deceased or sent to hospice from the dataset
#Some records with discharge disposition id = 11, 13, 14, 19, 20, 21 are removed since the patients cannot be readmitted to the hospital

diabetes <- subset(diabetes, discharge_disposition_id != 11 & discharge_disposition_id != 13 & 
                   discharge_disposition_id != 14 & discharge_disposition_id != 19 & discharge_disposition_id != 20 & discharge_disposition_id != 21)


## Change column type from numerical to categorical/nominal
#Several variables are in the formatted in wrong way. They are columns admission_type_id, admission_source_id, and discharge_disposition_id, as they are indicating ids. However, they were coded as numeric. Hence, all of them need to be converted to nominal/categorical variables.

diabetes$admission_type_id <- factor(diabetes$admission_type_id)
diabetes$admission_source_id <- factor(diabetes$admission_source_id)
diabetes$discharge_disposition_id <- factor(diabetes$discharge_disposition_id)


# Exploratory Data Analysis
## Handling Missing Values
### Finding columns containing missing values

colnames(diabetes)[colSums(is.na(diabetes)) > 0]

### race column

sum(is.na(diabetes['race']))/nrow(diabetes) * 100
sum(diabetes['race'] == 'Caucasian')/nrow(diabetes) * 100
levels(diabetes$race)[levels(diabetes$race) != 'Caucasian'] <- "Other"
diabetes$race[is.na(diabetes$race)] <- "Other"

#Race column has 2% missing data while the Caucasian race is 75% of the data. Hence, all other levels including NA are imputed to a new level called Other.

### weight column
sum(is.na(diabetes['weight']))/nrow(diabetes) * 100
diabetes <- subset(diabetes, select = -c(weight))

#Weight column has 97% missing data, so it will be removed from the analysis.

### payer_code column

sum(is.na(diabetes['payer_code']))/nrow(diabetes) * 100
diabetes <- subset(diabetes, select = -c(payer_code))

#Payer code column has almost 40% missing data and the columns deemed to be irrelevant to the analysis, so it will be removed.

## Detecting columns with low to zero variance

low_var_features <- nearZeroVar(diabetes, names = T, freqCut = 19, uniqueCut = 10)
low_var_features


## Removing columns with zero variance
#remove examide as it has zero variance
diabetes <- subset(diabetes, select = -c(examide))
#remove citoglipton as it has zero variance
diabetes <- subset(diabetes, select = -c(citoglipton))

# Feature Engineering
## Change columns level
### medical specialty column
sum(is.na(diabetes['medical_specialty']))/nrow(diabetes) * 100
diabetes$medical_specialty <- factor(diabetes$medical_specialty, levels=c(levels(diabetes$medical_specialty), "Missing"))
diabetes$medical_specialty[is.na(diabetes$medical_specialty)] <- "Missing"

levels(diabetes$medical_specialty)[levels(diabetes$medical_specialty) == "Family/GeneralPractice"] <- "General"
levels(diabetes$medical_specialty)[levels(diabetes$medical_specialty) %in% c("Cardiology", "Cardiology-Pediatric", "Gastroenterology", "Endocrinology", "Endocrinology-Metabolism", "Hematology", "Hematology/Oncology", "InternalMedicine", "Nephrology", "InfectiousDiseases", "Oncology", "Proctology", "Pulmonology", "Rheumatology", "SportsMedicine", "Urology")] <- "InternalMedicine"
levels(diabetes$medical_specialty)[levels(diabetes$medical_specialty) == "Emergency/Trauma"] <- "Emergency"
levels(diabetes$medical_specialty)[levels(diabetes$medical_specialty) %in% c("Anesthesiology", "Anesthesiology-Pediatric", "AllergyandImmunology", "Dentistry", "Dermatology", "Neurology", "Neurophysiology", "Ophthalmology", "Pathology", "Pediatrics", "Pediatrics-AllergyandImmunology", "Pediatrics-CriticalCare", "Pediatrics-EmergencyMedicine", "Pediatrics-Endocrinology", "Pediatrics-Hematology-Oncology", "Pediatrics-InfectiousDiseases", "Pediatrics-Neurology", "Pediatrics-Pulmonology", "Perinatology", "PhysicalMedicineandRehabilitation", "PhysicianNotFound", "Podiatry", "Psychiatry", "Psychiatry-Addictive", "Psychiatry-Child/Adolescent", "Psychology", "Radiologist", "Radiology", "Resident", "Speech", "Gynecology", "Obsterics&Gynecology-GynecologicOnco", "Obstetrics", "ObstetricsandGynecology", "OutreachServices", "DCPTEAM", "Hospitalist")] <- "Other"
levels(diabetes$medical_specialty)[levels(diabetes$medical_specialty) %in% c("Orthopedics", "Orthopedics-Reconstructive", "Osteopath", "Otolaryngology", "Surgeon", "Surgery-Cardiovascular", "Surgery-Cardiovascular/Thoracic", "Surgery-Colon&Rectal", "Surgery-General", "Surgery-Maxillofacial", "Surgery-Neuro", "Surgery-Pediatric", "Surgery-Plastic", "Surgery-PlasticwithinHeadandNeck", "Surgery-Thoracic", "Surgery-Vascular", "SurgicalSpecialty")] <- "Surgery"

#Medical Specialty column has 49% missing data, but it is kept since it probably has significance on the prediction. Hence, the missing values are recoded to a new level Missing. Moreover, medical specialty categories are grouped based on https://www.sgu.edu/blog/medical/ultimate-list-of-medical-specialties/ to simplify analysis process.

### admission columns
# Group and Recode Admission Source to Other and Emergency Category
levels(diabetes$admission_source_id)[levels(diabetes$admission_source_id) != '7'] <- 'Other'
levels(diabetes$admission_source_id)[levels(diabetes$admission_source_id) == '7'] <- 'Emergency'
levels(diabetes$admission_source_id)

#Grouping and Recoding Admission Source id to Emergency and Other, since the Emergency is half of the number of observations and the other values have similarities.

### discharge_disposition_id columns
# Group and Recode Discharge to Other and Home Category
levels(diabetes$discharge_disposition_id)[levels(diabetes$discharge_disposition_id) != '1'] <- 'Other'
levels(diabetes$discharge_disposition_id)[levels(diabetes$discharge_disposition_id) == '1'] <- 'Home'
levels(diabetes$discharge_disposition_id)

#Grouping and Recoding Discharge Source id to Home and Other, since the Emergency is half of the number of observations and the other values have similarities.

### admission_type_id columns
# Group and Recode Admission Source to Other and Emergency Category
levels(diabetes$admission_type_id)[levels(diabetes$admission_type_id) != '1'] <- 'Other'
levels(diabetes$admission_type_id)[levels(diabetes$admission_type_id) == '1'] <- 'Emergency'
levels(diabetes$admission_type_id)

# Change the column names to be more representative
setnames(diabetes, old=c('admission_type_id', 'discharge_disposition_id', 'admission_source_id'), new=c('admission_type', 'discharge_disposition', 'admission_source'))

#Grouping and Recoding Admission Type id to Emergency and Other, since the Emergency is half of the number of observations and the other values have similarities and change their name to be more representative.

### diagnosis columns
#summary(diabetes$diag_1)
sum(is.na(diabetes['diag_1']))/nrow(diabetes) * 100
#summary(diabetes$diag_2)
sum(is.na(diabetes['diag_2']))/nrow(diabetes) * 100
#summary(diabetes$diag_3)
sum(is.na(diabetes['diag_3']))/nrow(diabetes) * 100

`%notin%` <- Negate(`%in%`)
# Group and Recode Primary Diagnoses Result (diag_1)
levels(diabetes$diag_1)[levels(diabetes$diag_1) %notin% as.factor(c(390:459, 785, 460:519, 786, 520:579, 787, seq(250,250.99, 0.01), 800:999, 710:739, 580:629, 788, 140:239))] <- "Other"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(390:459, 785))] <- "Circulatory"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(460:519, 786))] <- "Respiratory"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(520:579, 787))] <- "Digestive"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(seq(250,250.99, 0.01)))] <- "Diabetes"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(800:999))] <- "Injury"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(710:739))] <- "Musculoskeletal"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(580:629, 788))] <- "Genitourinary"
levels(diabetes$diag_1)[levels(diabetes$diag_1) %in% as.factor(c(140:239))] <- "Neoplasms"
levels(diabetes$diag_1)

# Group and Recode Secondary Diagnoses Result (diag_2)
levels(diabetes$diag_2)[levels(diabetes$diag_2) %notin% as.factor(c(390:459, 785, 460:519, 786, 520:579, 787, seq(250,250.99, 0.01), 800:999, 710:739, 580:629, 788, 140:239))] <- "Other"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(390:459, 785))] <- "Circulatory"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(460:519, 786))] <- "Respiratory"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(520:579, 787))] <- "Digestive"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(seq(250,250.99, 0.01)))] <- "Diabetes"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(800:999))] <- "Injury"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(710:739))] <- "Musculoskeletal"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(580:629, 788))] <- "Genitourinary"
levels(diabetes$diag_2)[levels(diabetes$diag_2) %in% as.factor(c(140:239))] <- "Neoplasms"
levels(diabetes$diag_2)

# Group and Recode Secondary Additional Diagnoses Result (diag_3)
levels(diabetes$diag_3)[levels(diabetes$diag_3) %notin% as.factor(c(390:459, 785, 460:519, 786, 520:579, 787, seq(250,250.99, 0.01), 800:999, 710:739, 580:629, 788, 140:239))] <- "Other"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(390:459, 785))] <- "Circulatory"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(460:519, 786))] <- "Respiratory"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(520:579, 787))] <- "Digestive"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(seq(250,250.99, 0.01)))] <- "Diabetes"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(800:999))] <- "Injury"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(710:739))] <- "Musculoskeletal"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(580:629, 788))] <- "Genitourinary"
levels(diabetes$diag_3)[levels(diabetes$diag_3) %in% as.factor(c(140:239))] <- "Neoplasms"
levels(diabetes$diag_3)

#diag_1 only has 0.02 missing data, diag_2 has 0.4 missing data and diag_3 has 1.4 missing data, so they will be kept for further analysis.
#Diagnoses are grouped based on the paper Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014. Then, the rest of missing data are imputed using missForest package.

## Group the rest of diabetes medications into a new column called num_med and num_changes

keys <- c('metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin', 'glyburide.metformin', 'tolazamide', 'metformin.pioglitazone','metformin.rosiglitazone', 'glimepiride.pioglitazone', 'glipizide.metformin', 'troglitazone', 'tolbutamide', 'acetohexamide')

diabetes$num_med <- 0
diabetes$num_changes <- 0
for(key in keys){
  diabetes$num_med <- ifelse(diabetes[key] != 'No', diabetes$num_med + 1, diabetes$num_med)
  diabetes$num_changes <- ifelse((diabetes[key] == 'Up' | diabetes[key] == 'Down'), diabetes$num_changes + 1, diabetes$num_changes)
}
#summary(diabetes.test$num_changes)
#diabetes.fit$num_med <-factor(diabetes.fit$num_med)
#diabetes.fit$num_changes <-factor(diabetes.fit$num_changes)
diabetes <- subset(diabetes, select = -c(metformin, repaglinide, nateglinide, chlorpropamide, glimepiride, glipizide, glyburide, pioglitazone, rosiglitazone, acarbose, miglitol, insulin, glyburide.metformin, tolazamide, metformin.pioglitazone,metformin.rosiglitazone, glimepiride.pioglitazone, glipizide.metformin, troglitazone, tolbutamide, acetohexamide))
#colnames(diabetes.fit)
#diabetes2[,keys]

#The majority diabetes medications have low variance but it is unwise to drop the columns. Hence, to preserve the information, two new features generated from the diabetes medication columns

## Normalize, Remove Outliers, and Standardize Numerical Features
diabetes$number_inpatient <- log1p(diabetes$number_inpatient)
diabetes$number_outpatient <- log1p(diabetes$number_outpatient)
diabetes$number_emergency <- log1p(diabetes$number_emergency)

histogram(diabetes$number_inpatient)
histogram(diabetes$number_outpatient)
histogram(diabetes$number_emergency)

non_outliers = function(x, zs) {
  temp <- (x - mean(x))/sd(x)
  return(temp < zs)
}

diabetes <- diabetes[non_outliers(diabetes$number_inpatient, 3),]
diabetes <- diabetes[non_outliers(diabetes$number_outpatient, 3),]
diabetes <- diabetes[non_outliers(diabetes$number_emergency, 3),]
diabetes <- subset(diabetes, select = -c(number_emergency))

#Normalise skewed features and removing outliers using z-score

cols <- dplyr::select_if(diabetes, is.numeric)
temp <- scale(dplyr::select_if(diabetes, is.numeric))
for(col in colnames(cols)){
  diabetes[,col] <- temp[,col]
}

#Standardize numeric features

## Splitting data and impute the missing value
set.seed(1)
inTrain <- createDataPartition(diabetes$readmitted, p = .8)[[1]]

#diabetes.imp <- missForest(diabetes)

diabetes_train <- diabetes[inTrain,]
diabetes_test <- diabetes[-inTrain,]

head(diabetes_train)

diabetes_train.imp <- missForest(diabetes_train)
diabetes_test.imp <- missForest(diabetes_test)

#diabetes.imp$ximp
#diabetes2 <- data.frame(diabetes.imp$ximp)
diabetes_train2 <- data.frame(diabetes_train.imp$ximp)
diabetes_test2 <- data.frame(diabetes_test.imp$ximp)

## Upsampling and Downsampling the classes

diabetes_train_smoted <- SMOTE(readmitted ~ ., diabetes_train2, perc.over = 200, perc.under = 500)
diabetes_train_smoted2 <- SMOTE(readmitted ~ ., diabetes_train2, perc.over = 300, perc.under = 200)
summary(diabetes_train_smoted$readmitted)
summary(diabetes_test2$readmitted)

# Classification Algorithms
## Logistic Regression for prediction using GLM package and Lasso as well as for features selection
grid <- 10^seq(8,-2, length=100)
x.train <- model.matrix(readmitted ~ ., diabetes_train_smoted)
y.train <- diabetes_train_smoted$readmitted
x.test <- model.matrix(readmitted ~ ., diabetes_test2)
y.test <- diabetes_test2$readmitted

glmnet_model_cv <- cv.glmnet(x = x.train, y = y.train, family = "multinomial", alpha = 1, lambda = grid, standardize = T, nfolds = 5)

bestlam <- glmnet_model_cv$lambda.min

glmnet_model_2 <- glmnet(x = x.train, y = y.train, family = "multinomial", alpha = 1, lambda = bestlam, standardize = T)

glmnet.pred <- predict(glmnet_model_2, newx=x.test, s=bestlam, type="class")
acc <- sum(glmnet.pred == y.test) / length(y.test) * 100


glmnet.pred <- as.factor(glmnet.pred)
confussion_matrix_glmnet <- confusionMatrix(glmnet.pred, y.test)
print(confussion_matrix_glmnet)

non_zero_coef <- coef(glmnet_model_cv, s=bestlam)

#Accuracy of prediction with the best Lambda
#print(acc)

# Features with non-zero coefficient on Lasso with best lambda
#sort(non_zero_coef$NO@Dimnames[[1]][which(non_zero_coef$NO != 0)])
#sort(non_zero_coef$'<30'@Dimnames[[1]][which(non_zero_coef$'<30' != 0)])
#sort(non_zero_coef$'>30'@Dimnames[[1]][which(non_zero_coef$'>30' != 0)])

## GBM
set.seed(100)
gbm.model1 = gbm(readmitted ~ . , data = diabetes_train_smoted, distribution = "multinomial", n.trees = 1500,shrinkage = 0.1, interaction.depth = 5, cv.folds = 3)

test.label = diabetes_test2$readmitted
test.data = subset(diabetes_test2, select = -c(readmitted))

ntree_oob <- gbm.perf(gbm.model1, method = "OOB")

print(ntree_oob)

ntree_cv <- gbm.perf(gbm.model1, method = "cv")

print(ntree_cv)

pred = predict.gbm(object = gbm.model1,
                   newdata = test.data,
                   n.trees = 147,
                   type = "response")


pred_labels = colnames(pred)[apply(pred, 1, which.max)]

pred_labels = as.factor(pred_labels) 

sum(pred_labels == test.label) / length(test.label) * 100

gbm_conf_matrix = confusionMatrix(test.label, pred_labels)
print(gbm_conf_matrix )

###########################################################

pred = predict.gbm(object = gbm.model1,
                   newdata = test.data,
                   n.trees = 1500,
                   type = "response")


pred_labels = colnames(pred)[apply(pred, 1, which.max)]

pred_labels = as.factor(pred_labels) 

sum(pred_labels == test.label) / length(test.label) * 100

gbm_conf_matrix = confusionMatrix(test.label, pred_labels)
print(gbm_conf_matrix)

# Reduce shrinkage to 0.01 while keeping number of tree constant at 3200
#set.seed(100)
#gbm.model2 = gbm(readmitted ~ . , data = diabetes_train_smoted, distribution = "multinomial", n.trees = 3200,shrinkage = 0.01, interaction.depth = 5, #cv.folds = 3)

#ntree_oob2 <- gbm.perf(gbm.model2, method = "OOB")
#print(ntree_oob2)

#ntree_cv2 <- gbm.perf(gbm.model2, method = "cv")
#print(ntree_cv2)

#pred2 = predict.gbm(object = gbm.model2,
#                   newdata = test.data,
#                   n.trees = ntree_oob2,
#                   type = "response")


#pred_labels2 = colnames(pred2)[apply(pred2, 1, which.max)]

#pred_labels2 = as.factor(pred_labels2) 

#sum(pred_labels == test.label) / length(test.label) * 100

#gbm_conf_matrix2 = confusionMatrix(as.factor(test.label), pred_labels2)
#print(gbm_conf_matrix )


## Prediction using XGBoost
library(xgboost)
diabetes_train.fit3 <- data.frame(diabetes_train_smoted2)
diabetes_test.fit3 <- data.frame(diabetes_test2)

x_train <- model.matrix(readmitted ~ ., diabetes_train.fit3)
y_train <- diabetes_train.fit3$readmitted
x_test <- model.matrix(readmitted ~ ., diabetes_test.fit3)
y_test <- diabetes_test.fit3$readmitted

#md = 10
eta = 0.1
numberOfClasses <- length(unique(y_train))
#xgb_params <- list("objective" = "multi:softprob",
#                   "eval_metric" = "mlogloss",
#                   "num_class" = numberOfClasses,
#                   eta = eta,
#                   max_depth = 5, subsample=0.9, min_child_weight=4, colsample_bytree=0.2)

y_train <- as.numeric(y_train) - 1
y_test <- as.numeric(y_test) - 1

#nround    <- 10000 # number of XGBoost rounds
#cv.nfold  <- 5

# Begin - parameters tuning
#cv_model <- xgb.cv(params = xgb_params,
#                   data = x_train, 
#                   label = y_train,
#                   nrounds = nround,
#                   nfold = cv.nfold,
#                   verbose = FALSE,
#                   prediction = TRUE,
#                   early_stopping_rounds = 100)

#OOF_prediction <- data.frame(cv_model$pred) %>%
#  mutate(max_prob = max.col(., ties.method = "last"),
#         label = y_train + 1)
#head(OOF_prediction)

#confusionMatrix(factor(OOF_prediction$max_prob),
#                factor(OOF_prediction$label),
#                mode = "everything")
# End - parameters tuning

# Begin - Model Training and Prediction
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses,
                   eta = eta,
                   scale_pos_weight = 11,
                   max_depth = 3, subsample=0.9, min_child_weight=5, colsample_bytree=0.2)

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = x_test, label = y_test)

watchlist <- list(train=dtrain,test=dtest)
diabetes.xgb <- xgb.train(params = xgb_params, data=dtrain, nrounds = 10000, watchlist = watchlist, early_stopping_rounds = 100)
xgb.pred <- predict(diabetes.xgb, newdata = x_test, reshape = T)
xgb.pred <- as.data.frame(xgb.pred)
colnames(xgb.pred) <- levels(diabetes_test.fit3$readmitted)

xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(diabetes_test.fit3$readmitted)[y_test+1]

head(diabetes_test.fit3$readmitted)

head(xgb.pred$label)

head(xgb.pred$prediction,50)

confusionMatrix(factor(xgb.pred$prediction), factor(xgb.pred$label))