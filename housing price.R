################################################################################

#        S.Yang    Dec.08 2017

################################################################################
#
# PCA w/ regression and neural nets
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
################################################################################
# set memory limits
options(java.parameters = "-Xmx64048m") #64048 is 64 GB
################################################################################
# load data
################################################################################
# note that the features (1stFlrSF,2ndFlrSF) will get automatically renamed to 
#("X1stFlrSF","X2ndFlrSF") because names in R cannot begin with numbers

setwd("/Users/yangshenyang/Desktop/course/data mining/exam")

tr <- read.table("train.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2)
                             ,"numeric")
                )
te <- read.table("test.csv", header=T, sep=",", quote="",
                colClasses=c("numeric",rep("factor",2),rep("numeric",2),rep("factor",12)
                             ,rep("numeric",4),rep("factor",5),"numeric",rep("factor",7)
                             ,"numeric","factor",rep("numeric",3),rep("factor",4)
                             ,rep("numeric",10),"factor","numeric","factor"
                             ,"numeric",rep("factor",2),"numeric","factor"
                             ,rep("numeric",2),rep("factor",3),rep("numeric",6)
                             ,rep("factor",3),rep("numeric",3),rep("factor",2))
                )

################################################################################
# EDA
################################################################################
##percent of complete records 

source("DataQualityReportOverall.R")
DataQualityReportOverall(tr)
#   CompleteCases IncompleteCases CompleteCasePct
# 1             0            1460               0

### none of the cases is compelte, 
### i.e. 0% of the record is complete


# DataQualityReportOverall(te)
# # CompleteCases IncompleteCases CompleteCasePct
# # 1             0            1459               0

# percent of complete records 

source("DataQualityReport.R")
report <- DataQualityReport(tr)
missing <- report[report$NumberMissing != 0,]
missing$Attributes

# [1] LotFrontage  Alley        MasVnrType   MasVnrArea   BsmtQual     BsmtCond     BsmtExposure BsmtFinType1
# [9] BsmtFinType2 Electrical   FireplaceQu  GarageType   GarageYrBlt  GarageFinish GarageQual   GarageCond  
# [17] PoolQC       Fence        MiscFeature 

### There are 19 features that have missing values

################################################################################
# Preprocess data
################################################################################
#delete the features that have more than 80% of their values missing

moreThan80 <- missing[missing$PercentComplete < 20, ]
moreThan80$Attributes
#[1] Alley       PoolQC      Fence       MiscFeature

tr1 <- tr
te1 <- te

tr1$Id <- NULL
tr1$Alley <- NULL
tr1$PoolQC <- NULL
tr1$Fence <- NULL
tr1$MiscFeature <- NULL

te1$Alley <- NULL
te1$PoolQC <- NULL
te1$Fence <- NULL
te1$MiscFeature <- NULL

dim(tr)
# 1460   81
dim(tr1)
# 1460   76
dim(te) 
# 1459   80
dim(te1)
# 1459   76

# impute the records with missing values
library(mice)
tri0 <- mice(data=tr1, method='cart', seed=2016, maxint=1, printFlag=F)
tei0 <- mice(data=te1, method='cart', seed=2016, maxint=1, printFlag=F,threshold=1)

tri0 <- complete(tri0, action=1)
tei0 <- complete(tei0, action=1)

tri <- tri0
tei <- tei0

DataQualityReportOverall(tri)
#   CompleteCases IncompleteCases CompleteCasePct
# 1          1460               0             100

DataQualityReportOverall(tei)
#   CompleteCases IncompleteCases CompleteCasePct
# 1          1457               2           99.86
DataQualityReport(tei)
# 9      Utilities  factor             2           99.86 


################################################################################
# Zero- and Near Zero-Variance Predictors

library(caret)
dim(tri) 
# 1460   76

nearZeroVar <- nearZeroVar(tri,uniqueCut=3,names=FALSE,saveMetrics=FALSE)
tri_filtered <- tri[ ,-nearZeroVar]
tri <- cbind(tri$SalePrice, tri_filtered)
names(tri)[1] <- "SalePrice"
rm(tri_filtered)

# keep features in tei that were kept in tri
tei <- tei[,c("Id",names(tri)[2:(ncol(tri)-1)])]
dim(tri)
# 1460   60
dim(tei)
# 1459   59


################################################################################
# Creating Dummy Variables

# create dummies on tri 
names(Filter(is.factor, tri))
# [1] "MSSubClass"    "MSZoning"      "LotShape"      "LotConfig"     "Neighborhood"  "Condition1"    "BldgType"     
# [8] "HouseStyle"    "RoofStyle"     "Exterior1st"   "Exterior2nd"   "MasVnrType"    "ExterQual"     "ExterCond"    
# [15] "Foundation"    "BsmtQual"      "BsmtExposure"  "BsmtFinType1"  "HeatingQC"     "CentralAir"   "Electrical"   
# [22] "KitchenQual"   "FireplaceQu"   "GarageType"    "GarageFinish"  "PavedDrive"    "SaleType"   "SaleCondition"
### tri has 28 factor variables

dummies <- dummyVars( SalePrice ~ ., data = tri)
dmy <- data.frame(predict(dummies, newdata = tri))
names(dmy) <- gsub("\\.", "", names(dmy))
tri <- cbind(tri$SalePrice,dmy)
names(tri)[1] <- "SalePrice"

dim(tri)
# 1460  225

# create dummies on tei
dummies1 <- dummyVars( ~ ., data = tei)
dmy1 <- data.frame(predict(dummies1, newdata = tei))
names(dmy1) <- gsub("\\.", "", names(dmy1)) # removes dots from col names
tei <- dmy1
dim(tei)
# 1459  221
rm(dummies,dummies1, dmy,dmy1)

# ensure only features available in both train and score sets are kept
tri <- tri[,c("SalePrice", Reduce(intersect, list(names(tri), names(tei))))]
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

dim(tri)
# 1460  220
dim(tei)
# 1459  220

################################################################################
# Identify Correlated Predictors

# descrCor <-  cor(tri[,2:ncol(tri)])  # correlation matrix
# highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999) # number of features having a correlation greater than some value
# summary(descrCor[upper.tri(descrCor)])  # summarize the correlations
# # Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# # -1.0000000 -0.0352867 -0.0087267 -0.0002472  0.0306982  1.0000000 


descrCor <-  cor(tri[,2:ncol(tri)])  # correlation matrix
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.90)
filteredDescr <- tri[,2:ncol(tri)][,-highlyCorDescr] # remove those specific columns from your dataset
descrCor2 <- cor(filteredDescr) # calculate a new correlation matrix

summary(descrCor2[upper.tri(descrCor2)])
# Min.    1st Qu.     Median       Mean    3rd Qu.       Max. 
# -0.8945668 -0.0340700 -0.0085580  0.0002896  0.0300369  0.8832714 

### no correlaton higher than 0.9 

tri <- cbind(tri$SalePrice, filteredDescr)
names(tri)[1] <- "SalePrice"
rm(filteredDescr, descrCor, descrCor2, highlyCorDescr)

# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

dim(tri)
# 1460  206
dim(tei)
# 1459  206

################################################################################
# Identifying Linear Dependencies

comboInfo <- findLinearCombos(tri)
comboInfo[2]$remove
#  [1]  30  55  64  66 109 118 124 128 132 138 142 147 164 171 177 181 186 206
### the above columns are not independent, they need to be removed

tri <- tri[ ,-comboInfo[2]$remove]
rm(comboInfo)

# ensure same features show up in scoring set
tei <- tei[,c("Id", Reduce(intersect, list(names(tri), names(tei))))]

dim(tri)
# 1460  188
dim(tei)
# 1459  188

################################################################################
# make sure they are defined as factors. 

str(tri)
names(tri)
cols <- c("Id","MSSubClass120","MSSubClass160","MSSubClass180","MSSubClass190",
"MSSubClass20","MSSubClass30","MSSubClass40","MSSubClass45","MSSubClass50",
"MSSubClass60","MSSubClass70","MSSubClass75","MSSubClass80","MSSubClass85",
"MSSubClass90","MSZoningCall","MSZoningFV","MSZoningRH","MSZoningRL","MSZoningRM",
"StreetGrvl","StreetPave","AlleyGrvl","AlleyPave","LotShapeIR1","LotShapeIR2",
"LotShapeIR3","LotShapeReg","LandContourBnk","LandContourHLS","LandContourLow",
"LandContourLvl","UtilitiesAllPub","UtilitiesNoSeWa","LotConfigCorner","LotConfigCulDSac",
"LotConfigFR2","LotConfigFR3","LotConfigInside","LandSlopeGtl","LandSlopeMod",
"LandSlopeSev","NeighborhoodBlmngtn","NeighborhoodBlueste","NeighborhoodBrDale",
"NeighborhoodBrkSide","NeighborhoodClearCr","NeighborhoodCollgCr","NeighborhoodCrawfor",
"NeighborhoodEdwards","NeighborhoodGilbert","NeighborhoodIDOTRR","NeighborhoodMeadowV",
"NeighborhoodMitchel","NeighborhoodNAmes","NeighborhoodNoRidge","NeighborhoodNPkVill",
"NeighborhoodNridgHt","NeighborhoodNWAmes","NeighborhoodOldTown","NeighborhoodSawyer",
"NeighborhoodSawyerW","NeighborhoodSomerst","NeighborhoodStoneBr","NeighborhoodSWISU",
"NeighborhoodTimber","NeighborhoodVeenker","Condition1Artery","Condition1Feedr",
"Condition1Norm","Condition1PosA","Condition1PosN","Condition1RRAe","Condition1RRAn",
"Condition1RRNe","Condition1RRNn","Condition2Artery","Condition2Feedr","Condition2Norm",
"Condition2PosA","Condition2PosN","Condition2RRAe","Condition2RRAn","Condition2RRNn",
"BldgType1Fam","BldgType2fmCon","BldgTypeDuplex","BldgTypeTwnhs","BldgTypeTwnhsE",
"HouseStyle15Fin","HouseStyle15Unf","HouseStyle1Story","HouseStyle25Fin","HouseStyle25Unf",
"HouseStyle2Story","HouseStyleSFoyer","HouseStyleSLvl","RoofStyleFlat","RoofStyleGable",
"RoofStyleGambrel","RoofStyleHip","RoofStyleMansard","RoofStyleShed","RoofMatlClyTile",
"RoofMatlCompShg","RoofMatlMembran","RoofMatlMetal","RoofMatlRoll","RoofMatlTarGrv",
"RoofMatlWdShake","RoofMatlWdShngl","Exterior1stAsbShng","Exterior1stAsphShn",
"Exterior1stBrkComm","Exterior1stBrkFace","Exterior1stCBlock","Exterior1stCemntBd",
"Exterior1stHdBoard","Exterior1stImStucc","Exterior1stMetalSd","Exterior1stPlywood",
"Exterior1stStone","Exterior1stStucco","Exterior1stVinylSd","Exterior1stWdSdng",
"Exterior1stWdShing","Exterior2ndAsbShng","Exterior2ndAsphShn","Exterior2ndBrkCmn",
"Exterior2ndBrkFace","Exterior2ndCBlock","Exterior2ndCmentBd","Exterior2ndHdBoard",
"Exterior2ndImStucc","Exterior2ndMetalSd","Exterior2ndOther","Exterior2ndPlywood",
"Exterior2ndStone","Exterior2ndStucco","Exterior2ndVinylSd","Exterior2ndWdSdng",
"Exterior2ndWdShng","MasVnrTypeBrkCmn","MasVnrTypeBrkFace","MasVnrTypeNone",
"MasVnrTypeStone","ExterQualEx","ExterQualFa","ExterQualGd","ExterQualTA",
"ExterCondEx","ExterCondFa","ExterCondGd","ExterCondPo","ExterCondTA","FoundationBrkTil",
"FoundationCBlock","FoundationPConc","FoundationSlab","FoundationStone","FoundationWood",
"BsmtQualEx","BsmtQualFa","BsmtQualGd","BsmtQualTA","BsmtCondFa","BsmtCondGd",
"BsmtCondPo","BsmtCondTA","BsmtExposureAv","BsmtExposureGd","BsmtExposureMn",
"BsmtExposureNo","BsmtFinType1ALQ","BsmtFinType1BLQ","BsmtFinType1GLQ","BsmtFinType1LwQ",
"BsmtFinType1Rec","BsmtFinType1Unf","BsmtFinType2ALQ","BsmtFinType2BLQ","BsmtFinType2GLQ",
"BsmtFinType2LwQ","BsmtFinType2Rec","BsmtFinType2Unf","HeatingGasA","HeatingGasW",
"HeatingGrav","HeatingOthW","HeatingWall","HeatingQCEx","HeatingQCFa","HeatingQCGd",
"HeatingQCPo","HeatingQCTA","CentralAirN","CentralAirY","ElectricalFuseA",
"ElectricalFuseF","ElectricalFuseP","ElectricalMix","ElectricalSBrkr","KitchenQualEx",
"KitchenQualFa","KitchenQualGd","KitchenQualTA","FunctionalMaj1","FunctionalMaj2",
"FunctionalMin1","FunctionalMin2","FunctionalMod","FunctionalSev","FunctionalTyp",
"FireplaceQuEx","FireplaceQuFa","FireplaceQuGd","FireplaceQuPo","FireplaceQuTA",
"GarageType2Types","GarageTypeAttchd","GarageTypeBasment","GarageTypeBuiltIn",
"GarageTypeCarPort","GarageTypeDetchd","GarageFinishFin","GarageFinishRFn",
"GarageFinishUnf","GarageQualEx","GarageQualFa","GarageQualGd","GarageQualPo",
"GarageQualTA","GarageCondEx","GarageCondFa","GarageCondGd","GarageCondPo",
"GarageCondTA","PavedDriveN","PavedDriveP","PavedDriveY","PoolQCEx","PoolQCFa",
"PoolQCGd","FenceGdPrv","FenceGdWo","FenceMnPrv","FenceMnWw","MiscFeatureGar2",
"MiscFeatureOthr","MiscFeatureShed","MiscFeatureTenC","SaleTypeCOD","SaleTypeCon",
"SaleTypeConLD","SaleTypeConLI","SaleTypeConLw","SaleTypeCWD","SaleTypeNew",
"SaleTypeOth","SaleTypeWD","SaleConditionAbnorml","SaleConditionAdjLand",
"SaleConditionAlloca","SaleConditionFamily","SaleConditionNormal","SaleConditionPartial")
cols <- Reduce(intersect, list(names(tri), cols))
tri[cols] <- lapply(tri[cols], factor)
tei[cols] <- lapply(tei[cols], factor)

names(Filter(is.numeric, tri))
# [1] "SalePrice"     "LotFrontage"   "LotArea"       "OverallQual"   "OverallCond"   "YearBuilt"     "YearRemodAdd" 
# [8] "MasVnrArea"    "BsmtFinSF1"    "BsmtFinSF2"    "BsmtUnfSF"     "X1stFlrSF"     "X2ndFlrSF"     "GrLivArea"    
# [15] "BsmtFullBath"  "BsmtHalfBath"  "FullBath"      "HalfBath"      "BedroomAbvGr"  "TotRmsAbvGrd"  "Fireplaces"   
# [22] "GarageYrBlt"   "GarageCars"    "GarageArea"    "WoodDeckSF"    "OpenPorchSF"   "EnclosedPorch" "ScreenPorch"  
# [29] "MoSold"        "YrSold" 
names(Filter(is.numeric, tei))
# [1] "Id"            "LotFrontage"   "LotArea"       "OverallQual"   "OverallCond"   "YearBuilt"     "YearRemodAdd" 
# [8] "MasVnrArea"    "BsmtFinSF1"    "BsmtFinSF2"    "BsmtUnfSF"     "X1stFlrSF"     "X2ndFlrSF"     "GrLivArea"    
# [15] "BsmtFullBath"  "BsmtHalfBath"  "FullBath"      "HalfBath"      "BedroomAbvGr"  "TotRmsAbvGrd"  "Fireplaces"   
# [22] "GarageYrBlt"   "GarageCars"    "GarageArea"    "WoodDeckSF"    "OpenPorchSF"   "EnclosedPorch" "ScreenPorch"  
# [29] "MoSold"        "YrSold"


dim(tri)
# 1460  188
dim(tei)
# 1459  188

################################################################################
# standardize the input features 

standard <- preProcess(tri[names(tri)!='SalePrice'], method=c("range", "YeoJohnson"))
trit <- predict(standard,tri)

standard1 <- preProcess(tei[names(tei)!='ID'], method=c("range", "YeoJohnson"))
teit <- predict(standard1,tei)


dim(trit)
# [1] 1460  188
dim(teit)
# [1] 1459  188

### 187 features are left

nums <- sapply(trit, is.numeric)
trit_num <- trit[ ,nums]

nums2 <- sapply(teit, is.numeric)
teit_num <- teit[ ,nums2]

dim(trit_num)
# 1460   30
dim(teit_num)
# 1459   30

################################################################################
# Dimension Reduction - PCA
################################################################################

set.seed(1234)
index <- createDataPartition(trit_num$SalePrice, p=0.5, list=FALSE)
train <- trit_num[index,]
test <- trit_num[-index,]

train$SalePrice <- NULL
test$SalePrice <- NULL

dim(train)
# 731  29
dim(test)
# 729  29

library(psych)
pca <- principal(train
                  , nfactors = 29     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)


################          below is the pca using trit       ############################
# ### PCA will an error if there's facters in the df, 
# ### need to change the factors into num 
# str(train)
# ### 4 num features; 183 factors
# indx <- sapply(train, is.factor)
# train[indx] <- lapply(train[indx], function(x) as.numeric(as.character(x)))
# str(train)
# ### All num features
# Looking at the correlation among the numeric or integer variables
# corMatrix <- cor(train)
# #corMatrix
# cut_off <- 0.80
# # looping through the correlation matrix to identify multicollinear variables
# for (i in 1:dim(corMatrix)[1]) {
#   for (j in 1:dim(corMatrix)[2]) {
#     if(abs(corMatrix[i,j]) < cut_off | i==j | is.na(corMatrix[i,j] ) ) {
#       corMatrix[i,j] <- NA
#     }   else{
#       corMatrix[i,j] <- corMatrix[i,j]
#     }
#   }
# }
# train1 <- train
# train1$MSSubClass120 <- NULL
# train1$Exterior2ndCBlock <- NULL
# train1$ExterCondPo <- NULL
# train1$ElectricalFuseP <- NULL
# train1$SaleTypeOth <- NULL
# train$Exterior2ndCBlock <- NULL
# 
# # only show correlations that are "large" based off cut_off
# corMatrix <- corMatrix[, colSums(is.na(corMatrix)) < dim(corMatrix)[1]]
# corMatrix <- corMatrix[rowSums(is.na(corMatrix)) < dim(corMatrix)[2],]
# corMatrix
################         above is the pca using trit       ############################


pca$values
# [1] 6.72050608 3.17670256 2.55777013 1.83832805 1.33288598 1.17835382 1.10856492 1.04860255 0.98251024 0.96289450
# [11] 0.91686004 0.84908026 0.79896126 0.74527088 0.68402909 0.59963050 0.56489102 0.48284193 0.37502491 0.36131432
# [21] 0.33866134 0.29447751 0.25399140 0.21928856 0.20124916 0.16020687 0.12491324 0.10185013 0.02033874


pca$loadings
#                  PC1   PC2   PC3   PC4   PC5   PC6   PC7   PC8   PC9  PC10  PC11  PC12  PC13  PC14  PC15  PC16  PC17
# SS loadings    6.721 3.177 2.558 1.838 1.333 1.178 1.109 1.049 0.983 0.963 0.917 0.849 0.799 0.745 0.684 0.600 0.565
# Proportion Var 0.232 0.110 0.088 0.063 0.046 0.041 0.038 0.036 0.034 0.033 0.032 0.029 0.028 0.026 0.024 0.021 0.019
# Cumulative Var 0.232 0.341 0.429 0.493 0.539 0.579 0.618 0.654 0.688 0.721 0.753 0.782 0.809 0.835 0.859 0.879 0.899
#                 PC18  PC19  PC20  PC21  PC22  PC23  PC24  PC25  PC26  PC27  PC28  PC29
# SS loadings    0.483 0.375 0.361 0.339 0.294 0.254 0.219 0.201 0.160 0.125 0.102 0.020
# Proportion Var 0.017 0.013 0.012 0.012 0.010 0.009 0.008 0.007 0.006 0.004 0.004 0.001
# Cumulative Var 0.915 0.928 0.941 0.953 0.963 0.971 0.979 0.986 0.991 0.996 0.999 1.000

#### the first principal component accounts for 23.2% of the total variance 

plot(pca$values, type="b", main="Housing Data", col="blue")


# validation of pca on test 

pca1 <- principal(test
                  , nfactors = 9     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)
pca1$values
#  [1] 7.05845482 3.01513779 2.34498356 1.83421468 1.26524968 1.19527198 1.17195706 1.08791719 0.99665778
pca1$loadings
#                  PC1   PC2   PC3   PC4   PC5   PC6   PC7   PC8   PC9
# SS loadings    7.058 3.015 2.345 1.834 1.265 1.195 1.172 1.088 0.997
# Proportion Var 0.243 0.104 0.081 0.063 0.044 0.041 0.040 0.038 0.034
# Cumulative Var 0.243 0.347 0.428 0.491 0.535 0.576 0.617 0.654 0.689

plot(pca1$values, type="b", main="Scree plot for Housing Data", col="blue")

################################################################################
# Create PCs as input features to be used for prediction
################################################################################

pca_tr <- principal(trit_num[,2:ncol(trit_num)]
                  , nfactors = 29     # number of componets to extract
                  , rotate = "none"  # can specify different rotations
                  , scores = T       # find component scores or not
)

pca_te <- principal(teit_num[,2:ncol(teit_num)]
                 , nfactors = 29     # number of componets to extract
                 , rotate = "none"  # can specify different rotations
                 , scores = T       # find component scores or not
)

tr_pcscores <- data.frame(predict(pca_tr, data=trit_num[,2:ncol(trit_num)]))
te_pcscores <- data.frame(predict(pca_te, data=teit_num[,2:ncol(teit_num)]))

tr_pcscores <- tr_pcscores[,1:8]
te_pcscores <- te_pcscores[,1:8]

################################################################################
# Make sure datasets for all experiments are standardized similarly
################################################################################

# mix-max normalization 

preProcValues <- preProcess(tr_pcscores[,1:ncol(tr_pcscores)], method = c("range","YeoJohnson"))
tr_pcscores <- predict(preProcValues, tr_pcscores)
preProcValues <- preProcess(te_pcscores[,1:ncol(te_pcscores)], method = c("range","YeoJohnson"))
te_pcscores <- predict(preProcValues, te_pcscores)

facs <- sapply(trit, is.factor)
trit_facs <- trit[ ,facs]
tr_scoresNfactors <- data.frame(tr_pcscores, trit_facs)
facs2 <- sapply(teit, is.factor)
teit_facs <- teit[ ,facs2]
te_scoresNfactors <- data.frame(te_pcscores, teit_facs)

# ensure the target variables is in the new datasets
tr_pcscores <- data.frame(trit$SalePrice, tr_pcscores); names(tr_pcscores)[1] <- "SalePrice"
tr_scoresNfactors <- data.frame(trit$SalePrice, tr_scoresNfactors); names(tr_scoresNfactors)[1] <- "SalePrice"

dim(tr_pcscores)
# 1460    9
dim(te_pcscores)
# 1459    8
dim(tr_scoresNfactors)
# 1460  167
dim(te_scoresNfactors)
# 1459  166

################################################################################
# R Environment cleanup 
################################################################################
rm(imputedValues, imputedValues2, pca1, pca2, preProcValues,trainIndex, trit_facs
   , trit_num, facs, facs2, DataQualityReport, DataQualityReportOverall, pca_te
   , pca_tr, teit_facs, teit_num, train, test, pca, nums, nums2)


################################################################################
# Model building 
################################################################################

set.seed(1234)
library(caret)

trainIndex <- createDataPartition(trit$SalePrice # target variable vector
                                  , p = 0.80    # % of data for training
                                  , times = 1   # Num of partitions to create
                                  , list = F    # should result be a list (T/F)
)

# create a train and test set
train1 <- trit[trainIndex,]
test1 <- trit[-trainIndex,]

train2 <- tr_pcscores[trainIndex,]
test2 <- tr_pcscores[-trainIndex,]

train3 <- tr_scoresNfactors[trainIndex,]
test3 <- tr_scoresNfactors[-trainIndex,]


################################################################################
# multiple linear regression (forward and backward selection) on trit (train1) dataset
################################################################################

# forward selection
library(leaps)
mlf <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='forward') #plot(mlf)
vars2keep <- data.frame(summary(mlf)$which[which.max(summary(mlf)$adjr2),])
names(vars2keep) <- c("keep")  
head(vars2keep)
library(data.table)
vars2keep <- setDT(vars2keep, keep.rownames=T)[]
vars2keep <- c(vars2keep[which(vars2keep$keep==T & vars2keep$rn!="(Intercept)"),"rn"])[[1]]
vars2keep
# [1] "LotArea"              "NeighborhoodNoRidge1" "OverallQual"          "YearRemodAdd"         "BsmtQualEx1"         
# [6] "BsmtFinSF1"           "GrLivArea"            "KitchenQualEx1"       "GarageCars" 
modelFormula <- paste("SalePrice ~ LotArea + NeighborhoodNoRidge + OverallQual + YearRemodAdd + BsmtQualEx +
                                   BsmtFinSF1 + GrLivArea + KitchenQualEx + GarageCars")
m1f <- lm(modelFormula, data=train1)
train1$NeighborhoodNoRidge1
summary(m1f)
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -319363  -14442    -401   13177  245845 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)            -98374       6295 -15.628  < 2e-16 ***
#   LotArea                 95062      11694   8.129 1.10e-15 ***
#   NeighborhoodNoRidge1    67817       6586  10.297  < 2e-16 ***
#   OverallQual            167876      11576  14.502  < 2e-16 ***
#   YearRemodAdd            24905       3637   6.848 1.21e-11 ***
#   BsmtQualEx1             46525       4453  10.448  < 2e-16 ***
#   BsmtFinSF1              36517       4008   9.111  < 2e-16 ***
#   GrLivArea              163397      12140  13.460  < 2e-16 ***
#   KitchenQualEx1          43776       4820   9.082  < 2e-16 ***
#   GarageCars              55839       7681   7.270 6.61e-13 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 34750 on 1159 degrees of freedom
# Multiple R-squared:  0.8193,	Adjusted R-squared:  0.8179 
# F-statistic: 583.8 on 9 and 1159 DF,  p-value: < 2.2e-16



# backward selection
library(leaps)
mlb <- regsubsets(SalePrice ~ ., data=train1, nbest=1, intercept=T, method='backward') #plot(mlf)
vars2keep1 <- data.frame(summary(mlb)$which[which.max(summary(mlb)$adjr2),])
names(vars2keep1) <- c("keep")  
head(vars2keep1)
library(data.table)
vars2keep1 <- setDT(vars2keep1, keep.rownames=T)[]
vars2keep1 <- c(vars2keep1[which(vars2keep1$keep==T & vars2keep1$rn!="(Intercept)"),"rn"])[[1]]
vars2keep1
# [1] "LotArea"              "NeighborhoodNoRidge1" "OverallQual"          "BsmtQualEx1"          "BsmtExposureGd1"     
# [6] "BsmtFinSF1"           "GrLivArea"            "KitchenQualEx1"       "GarageCars"
modelFormula1 <- paste("SalePrice ~  LotArea + NeighborhoodNoRidge + OverallQual + BsmtQualEx + BsmtExposureGd +
                                    BsmtFinSF1 + GrLivArea + KitchenQualEx + GarageCars") 
m1b <- lm(modelFormula1, data=train1)
summary(m1b)
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -326134  -16550     928   15090  238404 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)            -93259       6432 -14.499  < 2e-16 ***
#   LotArea                 74428      11856   6.277 4.86e-10 ***
#   NeighborhoodNoRidge1    66764       6620  10.085  < 2e-16 ***
#   OverallQual            188232      11010  17.097  < 2e-16 ***
#   BsmtQualEx1             44810       4505   9.946  < 2e-16 ***
#   BsmtExposureGd1         21631       3756   5.759 1.08e-08 ***
#   BsmtFinSF1              30909       4124   7.494 1.32e-13 ***
#   GrLivArea              165704      12222  13.557  < 2e-16 ***
#   KitchenQualEx1          44391       4850   9.153  < 2e-16 ***
#   GarageCars              63995       7617   8.402  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 34950 on 1159 degrees of freedom
# Multiple R-squared:  0.8172,	Adjusted R-squared:  0.8158 
# F-statistic: 575.7 on 9 and 1159 DF,  p-value: < 2.2e-16

source("myDiag.R")
myDiag(m1f)
myDiag(m1b)

# plot predicted vs actual
par(mfrow=c(1,2))
yhat_m1f <- predict(m1f, newdata=train1); plot(train1$SalePrice, yhat_m1f)
yhat_m1b <- predict(m1b, newdata=train1); plot(train1$SalePrice, yhat_m1b)

################################################################################
# multiple linear regression on tr_pcscores (train2) dataset
################################################################################
library(caret)
m2 <- lm(SalePrice ~ ., data=train2)
summary(m2)
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -280970  -22612   -4467   16099  409150 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  -187981      14160 -13.276  < 2e-16 ***
#   PC1           452925       8439  53.668  < 2e-16 ***
#   PC2           -26803       8079  -3.317 0.000937 ***
#   PC3            80849      10074   8.026 2.46e-15 ***
#   PC4            23102       6931   3.333 0.000885 ***
#   PC5            29365      11249   2.610 0.009160 ** 
#   PC6            42955       8299   5.176 2.67e-07 ***
#   PC7            60480      10679   5.663 1.87e-08 ***
#   PC8            61554       9290   6.626 5.27e-11 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 42620 on 1160 degrees of freedom
# Multiple R-squared:  0.7279,	Adjusted R-squared:  0.726 
# F-statistic: 387.9 on 8 and 1160 DF,  p-value: < 2.2e-16

myDiag(m2)
# plot predicted vs actual
par(mfrow=c(1,1))
yhat_m2 <- predict(m2, newdata=train2); plot(train2$SalePrice, yhat_m2)

################################################################################
# multiple linear regression on tr_scoresNfactors (train3) dataset
################################################################################

m3 <- lm(SalePrice ~ ., data=train3)
summary(m3)
# Residual standard error: 31740 on 1003 degrees of freedom
# Multiple R-squared:  0.8695,	Adjusted R-squared:  0.848 
# F-statistic:  40.5 on 165 and 1003 DF,  p-value: < 2.2e-16

# backward selection
m3b <- regsubsets(SalePrice ~ ., data=train3, nbest=1, intercept=T, method='backward') #plot(mlf)
vars2keep2 <- data.frame(summary(m3b)$which[which.max(summary(m3b)$adjr2),])
names(vars2keep2) <- c("keep")  
head(vars2keep2)
library(data.table)
vars2keep2 <- setDT(vars2keep2, keep.rownames=T)[]
vars2keep2 <- c(vars2keep2[which(vars2keep2$keep==T & vars2keep2$rn!="(Intercept)"),"rn"])[[1]]
vars2keep2
# [1] "PC1"                  "PC3"                  "NeighborhoodNoRidge1" "NeighborhoodStoneBr1" "ExterQualEx1"        
# [6] "ExterQualGd1"         "BsmtQualEx1"          "BsmtExposureGd1"      "KitchenQualEx1"  
modelFormula1 <- paste("SalePrice ~ PC1 + PC3 + NeighborhoodNoRidge + NeighborhoodStoneBr + ExterQualEx +      
                                      + ExterQualGd + BsmtQualEx + BsmtExposureGd + KitchenQualEx") 
m3b <- lm(modelFormula1, data=train3)
summary(m3b)
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -335878  -16516    -852   13591  277755 
# 
# Coefficients:
#   Estimate Std. Error t value Pr(>|t|)    
# (Intercept)            -17364       5437  -3.194  0.00144 ** 
#   PC1                    300301      10108  29.709  < 2e-16 ***
#   PC3                     67894       9121   7.444 1.90e-13 ***
#   NeighborhoodNoRidge1    76806       6648  11.554  < 2e-16 ***
#   NeighborhoodStoneBr1    56050       8140   6.885 9.43e-12 ***
#   ExterQualEx1            43929       7779   5.647 2.05e-08 ***
#   ExterQualGd1            20396       3050   6.689 3.50e-11 ***
#   BsmtQualEx1             37686       4693   8.030 2.38e-15 ***
#   BsmtExposureGd1         25171       3754   6.705 3.14e-11 ***
#   KitchenQualEx1          46392       5249   8.838  < 2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# Residual standard error: 35300 on 1159 degrees of freedom
# Multiple R-squared:  0.8135,	Adjusted R-squared:  0.8121 
# F-statistic: 561.8 on 9 and 1159 DF,  p-value: < 2.2e-16
# perform regression diagnostics and discuss if you see any potential issues or 
# assumption violations.



# source("myDiag.R")
# myDiag(m3)
# # plot predicted vs actual
# par(mfrow=c(1,2))
# yhat_m3 <- predict(m3, newdata=train3); plot(train3$SalePrice, yhat_m3)
# yhat_m3b <- predict(m3b, newdata=train3); plot(train3$SalePrice, yhat_m3b)

################################################################################
# Neural Networks
################################################################################

library(caret)
ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=3         # k number of times to do k-fold
)

(maxvalue <- summary(trit$SalePrice)["Max."][[1]])
# 755000

nnet1 <- train(SalePrice/755000 ~ LotArea + NeighborhoodNoRidge + OverallQual + YearRemodAdd + BsmtQualEx 
                                + GrLivArea + KitchenQualEx + GarageCars + BsmtExposureGd + BsmtFinSF1,
                  data = train1,     # training set used to build model
                  method = "nnet",     # type of model you want to build
                  trControl = ctrl,    # how you want to learn
                  tuneLength = 15,
                  maxit = 100,
                  metric = "RMSE"     # performance measure
)

#summary(nnet1)

nnet1$finalModel$tuneValue
#    size        decay
# 48    7 0.0001701254

#add grid
myGrid <-  expand.grid(size = c(5,6,7,8,9)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.0001701254
                                   ,0.0101701254))  #parameter for weight decay. 

nnet1b <- train(SalePrice/755000 ~ LotArea + NeighborhoodNoRidge + OverallQual + YearRemodAdd + BsmtQualEx 
                                   + GrLivArea + KitchenQualEx + GarageCars + BsmtExposureGd + BsmtFinSF1,
               data = train1,     # training set used to build model 
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneGrid = myGrid,
               maxit = 500,
               metric = "RMSE"     # performance measure
)

par(mfrow=c(1,2))
yhat_nn1 <- predict(nnet1, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1)
yhat_nn1b <- predict(nnet1b, newdata=train1)*maxvalue; plot(train1$SalePrice, yhat_nn1b)

nnet1b$finalModel$tuneValue
#   size       decay
# 5    6 0.0001701254


nnet2 <- train(SalePrice/755000 ~ . ,
               data = train2,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE"     # performance measure
)

nnet2$finalModel$tuneValue
#    size      decay
# 63    9 0.0001701254


myGrid1 <-  expand.grid(size = c(7,8,9,10,11)     # number of units in the hidden layer.
                       , decay = c(0
                                   ,0.0001701254
                                   ,0.0101701254))  #parameter for weight decay. 
nnet2b <- train(SalePrice/755000 ~ .,
                data = train2,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid1,
                maxit = 500,
                metric = "RMSE"     # performance measure
)

nnet2b$finalModel$tuneValue
#    size      decay
# 15   7  0.01017013

par(mfrow=c(1,2))
yhat_nn2 <- predict(nnet2, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2)
yhat_nn2b <- predict(nnet2b, newdata=train2)*maxvalue; plot(train2$SalePrice, yhat_nn2b)

nnet3 <- train(SalePrice/755000 ~ PC1 + PC3 + NeighborhoodNoRidge + NeighborhoodStoneBr + ExterQualEx +      
                                  + ExterQualGd + BsmtQualEx + BsmtExposureGd + KitchenQualEx,
               data = train3,     # training set used to build model
               method = "nnet",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               tuneLength = 15,
               maxit = 100,
               metric = "RMSE"     # performance measure
)

nnet3$finalModel$tuneValue
#    size      decay
# 18    3 0.0001701254


myGrid2 <-  expand.grid(size = c(1,2,3,4,5)     # number of units in the hidden layer.
             , decay = c(0
                         ,0.0001701254
                         ,0.0101701254))  #parameter for weight decay. 

nnet3b <- train(SalePrice/755000 ~ PC1 + PC3 + NeighborhoodNoRidge + NeighborhoodStoneBr + ExterQualEx +      
                                    + ExterQualGd + BsmtQualEx + BsmtExposureGd + KitchenQualEx,
                data = train3,     # training set used to build model 
                method = "nnet",     # type of model you want to build
                trControl = ctrl,    # how you want to learn
                tuneGrid = myGrid2,
                maxit = 500,
                metric = "RMSE"     # performance measure
)


yhat_nn3 <- predict(nnet3, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3)
yhat_nn3b <- predict(nnet3b, newdata=train3)*maxvalue; plot(train3$SalePrice, yhat_nn3b)


################################################################################
# Decision Trees
################################################################################

library(tree)
tree1 = tree(SalePrice ~ .
               , control = tree.control(nobs=nrow(train1)[[1]]
                                        , mincut = 0
                                        , minsize = 1
                                        , mindev = 0.01)
               , data = train1)
summary(tree1)
# Regression tree:
#   tree(formula = SalePrice ~ ., data = train1, control = tree.control(nobs = nrow(train1)[[1]], 
#                                                                       mincut = 0, minsize = 1, mindev = 0.01))
# Variables actually used in tree construction:
#   [1] "OverallQual" "X1stFlrSF"   "GrLivArea"   "YearBuilt"   "X2ndFlrSF"   "GarageCars"  "WoodDeckSF" 
# Number of terminal nodes:  12 
# Residual mean deviance:  1.345e+09 = 1.556e+12 / 1157 
# Distribution of residuals:
#   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -185000.0  -20510.0     643.7       0.0   18940.0  190800.0 


# plot(tree1)
# text(tree1, pretty=0,cex=0.7) # plot the tree
# ### cex specifies the font of the rext


# perform cross-validation to find optimal number of terminal nodes
cv.tree1 = cv.tree(tree1)
par(mfrow=c(1,1))
plot(cv.tree1$size
     , cv.tree1$dev
     , type = 'b')


### prune tree where the number of terminal nodes is 6? 4?
prunedfit = prune.tree(tree1, best=6)
summary(prunedfit)
# Regression tree:
#   snip.tree(tree = tree1, nodes = c(10L, 4L, 13L, 11L, 12L))
# Variables actually used in tree construction:
#   [1] "OverallQual" "GrLivArea"   "X2ndFlrSF"  
# Number of terminal nodes:  6 
# Residual mean deviance:  1.846e+09 = 2.146e+12 / 1163 
# Distribution of residuals:
#   Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
# -215100.0  -25420.0    -122.9       0.0   20590.0  236600.0 
plot(prunedfit); text(prunedfit, pretty=0,cex=0.9)


yhat_tree1 <- predict(prunedfit, newdata=test1)
plot(yhat_tree1,test1$SalePrice)


## bagged tree on train1
tree1b <- train(SalePrice ~ .,
               data = train1,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

## bagged tree on train2
tree2 <- train(SalePrice ~ .,
               data = train2,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,
               metric = "RMSE"     # performance measure
)

## bagged tree on train3
tree3 <- train(SalePrice ~ .,
               data = train3,     # training set used to build model
               method = "treebag",     # type of model you want to build
               trControl = ctrl,    # how you want to learn
               metric = "RMSE"     # performance measure
)

par(mfrow=c(2,2))
yhat_dt1 <- predict(tree1, newdata=train1); plot(train1$SalePrice, yhat_dt1)
yhat_dt1b <- predict(tree1b, newdata=train1); plot(train1$SalePrice, yhat_dt1b)
yhat_dt2 <- predict(tree2, newdata=train2); plot(train2$SalePrice, yhat_dt2)
yhat_dt3 <- predict(tree3, newdata=train3); plot(train3$SalePrice, yhat_dt3)

################################################################################
# Model Evaluation
################################################################################
# Q40)
yhat_m1f_te <- predict(m1f, newdata=test1)
yhat_m1b_te <- predict(m1b, newdata=test1)
yhat_m2_te <- predict(m2, newdata=test2)
yhat_m3_te <- predict(m3, newdata=test3)
yhat_m3b_te <- predict(m3b, newdata=test3)

yhat_nn1_te <- predict(nnet1, newdata=test1)*maxvalue
yhat_nn1b_te <- predict(nnet1b, newdata=test1)*maxvalue
yhat_nn2_te <- predict(nnet2, newdata=test2)*maxvalue
yhat_nn2b_te <- predict(nnet2b, newdata=test2)*maxvalue
yhat_nn3_te <- predict(nnet3, newdata=test3)*maxvalue
yhat_nn3b_te <- predict(nnet3b, newdata=test3)*maxvalue

yhat_dt1_te <- predict(tree1, newdata=test1)
yhat_dt1b_te <- predict(tree1b, newdata=test1)
yhat_dt2_te <- predict(tree2, newdata=test2)
yhat_dt3_te <- predict(tree3, newdata=test3)

results <- matrix(rbind(
cbind(t(postResample(pred=yhat_m1f, obs=train1$SalePrice)), t(postResample(pred=yhat_m1f_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m1b, obs=train1$SalePrice)), t(postResample(pred=yhat_m1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_m2, obs=train2$SalePrice)), t(postResample(pred=yhat_m2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_m3, obs=train3$SalePrice)), t(postResample(pred=yhat_m3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_m3b, obs=train3$SalePrice)), t(postResample(pred=yhat_m3b_te, obs=test3$SalePrice))),
  
cbind(t(postResample(pred=yhat_nn1, obs=train1$SalePrice)), t(postResample(pred=yhat_nn1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn1b, obs=train1$SalePrice)), t(postResample(pred=yhat_nn1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_nn2, obs=train2$SalePrice)), t(postResample(pred=yhat_nn2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn2b, obs=train2$SalePrice)), t(postResample(pred=yhat_nn2b_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_nn3, obs=train3$SalePrice)), t(postResample(pred=yhat_nn3_te, obs=test3$SalePrice))),
cbind(t(postResample(pred=yhat_nn3b, obs=train3$SalePrice)), t(postResample(pred=yhat_nn3b_te, obs=test3$SalePrice))),
    
cbind(t(postResample(pred=yhat_dt1, obs=train1$SalePrice)), t(postResample(pred=yhat_dt1_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt1b, obs=train1$SalePrice)), t(postResample(pred=yhat_dt1b_te, obs=test1$SalePrice))),
cbind(t(postResample(pred=yhat_dt2, obs=train2$SalePrice)), t(postResample(pred=yhat_dt2_te, obs=test2$SalePrice))),
cbind(t(postResample(pred=yhat_dt3, obs=train3$SalePrice)), t(postResample(pred=yhat_dt3_te, obs=test3$SalePrice)))
), nrow=15)
colnames(results) <- c("Train_RMSE", "Train_R2","Train_MAE", "Test_RMSE", "Test_R2","Test_MAE")
rownames(results) <- c("MLR_Forward","MLR_Backward","MLR_PCs","MLR_PCs+Factors",
                       "MLR_Backward_PCs+Factors","NN_ForBackFeatures","NN_ForBackFeatures_Optimized",
                       "NN_PCs","NN_PCs_Optimized","NN_BackFeatures","NN_BackFeatures_Optimized",
                       "Tree_Numerics+Factors","BaggedTree_Numerics+Factors",
                       "BaggedTree_PCs","BaggedTree_PCs+Factors")
results
#                              Train_RMSE  Train_R2 Train_MAE Test_RMSE   Test_R2 Test_MAE
# MLR_Forward                    34600.97 0.8192741  21342.55  28706.42 0.8370950 21152.14
# MLR_Backward                   34799.74 0.8171918  22444.94  29446.23 0.8281109 21859.10
# MLR_PCs                        42456.02 0.7279040  27112.47  32803.42 0.7885794 24820.31
# MLR_PCs+Factors                29403.68 0.8694890  18298.27  25701.26 0.8697342 17714.83
# MLR_Backward_PCs+Factors       35146.06 0.8135351  22129.63  28905.06 0.8361563 20933.21
# NN_ForBackFeatures             29951.24 0.8647471  18606.18  25415.75 0.8718702 18286.58
# NN_ForBackFeatures_Optimized   22460.25 0.9238864  15854.92  26049.46 0.8657787 18235.15
# NN_PCs                         30304.00 0.8614488  19530.19  25917.63 0.8671659 18976.15
# NN_PCs_Optimized               35585.44 0.8089128  21143.31  25632.86 0.8695527 19137.24
# NN_BackFeatures                29684.35 0.8672202  20146.24  26510.92 0.8606347 18668.44
# NN_BackFeatures_Optimized      33954.64 0.8262776  20630.71  26547.07 0.8612647 18956.49
# Tree_Numerics+Factors          36488.01 0.7990241  26312.83  42216.46 0.6813463 30783.05
# BaggedTree_Numerics+Factors    32171.19 0.8539817  21678.07  29946.74 0.8274275 22360.44
# BaggedTree_PCs                 33842.54 0.8285497  21884.57  30800.05 0.8144634 22506.95
# BaggedTree_PCs+Factors         34098.95 0.8260610  21962.73  30422.08 0.8185511 22200.64

library(reshape2)
results <- melt(results)
names(results) <- c("Model","Stat","Values")


library(ggplot2)
# RMSE
p1 <- ggplot(data=results[which(results$Stat=="Train_RMSE" | results$Stat=="Test_RMSE"),]
            , aes(x=Model, y=Values, fill=Stat)) 
p1 <- p1 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p1 <- p1 + facet_grid(~Model, scale='free_x', drop = TRUE)
p1 <- p1 + scale_fill_manual(values=c('#FF6666','blue'))
p1 <- p1 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p1 <- p1 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
               strip.text.y = element_text(size=0, face="bold"),
               strip.background = element_rect(colour="white", fill="white"))
p1 <- p1 + ggtitle("RMSE Performance")
p1

# R2
# 
p1 <- ggplot(data=results[which(results$Stat=="Train_R2" | results$Stat=="Test_R2"),]
             , aes(x=Model, y=Values, fill=Stat)) 
p1 <- p1 + geom_bar(stat="identity", color="black", position=position_dodge()) + theme_minimal()
p1 <- p1 + facet_grid(~Model, scale='free_x', drop = TRUE)
p1 <- p1 + scale_fill_manual(values=c('#FF6666','blue'))
p1 <- p1 + xlab(NULL) + theme(axis.text.x = element_text(angle = 90, hjust=1, vjust=.5))

p1 <- p1 + theme(strip.text.x = element_text(size=0, angle=0, colour="white"),
                 strip.text.y = element_text(size=0, face="bold"),
                 strip.background = element_rect(colour="white", fill="white"))
p1 <- p1 + ggtitle("R2 Performance")
p1



################################################################################
# Score data / Deployment
################################################################################


yhat_final <- predict(nnet1b, newdata=teit)*maxvalue
submission <- read.table("sample_submission.csv",header=T, sep=",")
submission$SalePrice <- yhat_final

# Write out file to be uploaded to Kaggle.com for scoring
write.table(submission, file=paste0("submission_SY.csv") 
            , quote=F, sep=",", row.names=F, col.names=T)


