##################################
#    Author: Debra Nyanchama    #
##################################

##################################

cat("Cleaning memory\n");
rm(list = ls()); gc();

library(data.table);
library(xgboost);
library(Matrix);

target <- "readmitted";

train <- fread("C:/Users/semed/Music/dataset_diabetes/diabetic_data.csv", na.strings = "?");

#dropping because I cannot use the same user in differfent folds in the CV process.
#without dropping these values, your CV will improve. As we do not have timestamp information, I have decided to drop these two columns to be realistic.
train$encounter_id <- NULL;
train$patient_nbr <- NULL;

train$race <- as.numeric(as.factor(train$race));

train$age <- ifelse(train$age == "[0-10)",  0, train$age);
train$age <- ifelse(train$age == "[10-20)", 10, train$age);
train$age <- ifelse(train$age == "[20-30)", 20, train$age);
train$age <- ifelse(train$age == "[30-40)", 30, train$age);
train$age <- ifelse(train$age == "[40-50)", 40, train$age);
train$age <- ifelse(train$age == "[50-60)", 50, train$age);
train$age <- ifelse(train$age == "[60-70)", 60, train$age);
train$age <- ifelse(train$age == "[70-80)", 70, train$age);
train$age <- ifelse(train$age == "[80-90)", 80, train$age);
train$age <- ifelse(train$age == "[90-100)", 90, train$age);
train$age <- as.numeric(train$age);

train$gender <- as.numeric(as.factor(train$gender));

train$weight <- ifelse(train$weight == "[75-100)",  75, train$weight);
train$weight <- ifelse(train$weight == "[50-75)",   50, train$weight);
train$weight <- ifelse(train$weight == "[25-50)",   25, train$weight);
train$weight <- ifelse(train$weight == "[0-25)",    0, train$weight);
train$weight <- ifelse(train$weight == "[100-125)", 100, train$weight);
train$weight <- ifelse(train$weight == "[125-150)", 125, train$weight);
train$weight <- ifelse(train$weight == "[150-175)", 150, train$weight);
train$weight <- ifelse(train$weight == "[175-200)", 175, train$weight);
train$weight <- ifelse(train$weight == ">200",      -25, train$weight);
train$weight <- as.numeric(train$weight);

train$admission_type_id <- as.numeric(as.factor(train$admission_type_id));
train$discharge_disposition_id <- as.numeric(as.factor(train$discharge_disposition_id));
train$admission_source_id <- as.numeric(as.factor(train$admission_source_id));
train$time_in_hospital <- as.numeric(train$time_in_hospital);
train$payer_code <- as.numeric(as.factor(train$payer_code));
train$medical_specialty <- as.numeric(as.factor(train$medical_specialty));
train$num_lab_procedures <- as.numeric(train$num_lab_procedures);
train$num_procedures <- as.numeric(train$num_procedures);
train$num_medications <- as.numeric(train$num_medications);
train$number_outpatient <- as.numeric(train$number_outpatient);
train$number_emergency <- as.numeric(train$number_emergency);
train$number_inpatient <- as.numeric(train$number_inpatient);
train$diag_1 <- as.numeric(as.factor(train$diag_1));
train$diag_2 <- as.numeric(as.factor(train$diag_2));
train$diag_3 <- as.numeric(as.factor(train$diag_3));
train$number_diagnoses <- as.numeric(train$number_diagnoses);

train$max_glu_serum <- ifelse(train$max_glu_serum == "None",  0, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == "Norm",  100, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == ">200",  200, train$max_glu_serum);
train$max_glu_serum <- ifelse(train$max_glu_serum == ">300",  300, train$max_glu_serum);
train$max_glu_serum <- as.numeric(train$max_glu_serum);

train$A1Cresult <- ifelse(train$A1Cresult == "None",  0, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == "Norm",  5, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == ">7",    7, train$A1Cresult);
train$A1Cresult <- ifelse(train$A1Cresult == ">8",    8, train$A1Cresult);
train$A1Cresult <- as.numeric(train$A1Cresult);

columns <- c("metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone");
for( c in columns ){
  train[[c]] <- ifelse(train[[c]] == "Up",     +10, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "Down",   -10, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "Steady", +0, train[[c]]);
  train[[c]] <- ifelse(train[[c]] == "No",     -20, train[[c]]);
  train[[c]] <- as.numeric(train[[c]]);
}

train$change <- ifelse(train$change == "No", -1, train$change);
train$change <- ifelse(train$change == "Ch", +1, train$change);
train$change <- as.numeric(train$change);

train$diabetesMed <- ifelse(train$diabetesMed == "Yes", +1, train$diabetesMed);
train$diabetesMed <- ifelse(train$diabetesMed == "No",  -1, train$diabetesMed);
train$diabetesMed <- as.numeric(train$diabetesMed);

train$readmitted <- ifelse(train$readmitted != "NO", 1, 0); # ">30", "<30", "NO"
train$readmitted <- as.numeric(train$readmitted);

train[] <- lapply(train, as.numeric);

train.y <- train[[target]];
train[[target]] <- NULL;

gc();

dtrain <- xgb.DMatrix(as.matrix(train), label = train.y, missing = NA);
watchlist <- list(train = dtrain);

param <- list(
  objective           = "reg:logistic",
  booster             = "gbtree",
  eta                 = 0.03,
  max_depth           = 5,
  eval_metric         = "auc",
  min_child_weight    = 150,
  alpha               = 0.00,
  subsample           = 0.70,
  colsample_bytree    = 0.70
);

set.seed(1981);
clf <- xgb.cv(  params                = param,
                data                  = dtrain,
                nrounds               = 20000,
                verbose               = 1,
                watchlist             = watchlist,
                maximize              = TRUE,
                nfold                 = 5,
                nthread               = 4,
                print_every_n         = 50,
                stratified            = TRUE,
                early_stopping_rounds = 10
);
