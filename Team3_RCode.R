#Install necessary packages

install.packages("rpart.plot")
install.packages("naniar")
install.packages("BiocManager")
install.packages("ComplexHeatmap")

# Load required libraries for data manipulation, visualization, and modeling

library(naniar)
library(ggplot2)
library(tidyr)
library(dplyr)
library(caret)
library(VIM)
library(pROC)
library(e1071)
library(rpart)
library(rpart.plot)
library(ComplexHeatmap)
library(circlize)


# Load the dataset
data <- read.csv("C:\\Users\\Ravipati09\\Documents\\574\\Project\\Team3_Dataset.csv")

# Treat empty strings as NA
data[data == ""] <- NA

# Step 1: Check and visualize missing value patterns

# Calculate and print percentage of missing values for each column
namedCounts <- sapply(data, function(x) round((sum(is.na(x))/length(x))*100,2))
namedCounts <- namedCounts[namedCounts>0]
if (length(namedCounts) > 0) {
  cat("Columns with missing values:\n")
  for (name in names(namedCounts)) {
    cat(name, ":", namedCounts[name], "%\n")
  }
} else {
  cat("No columns with missing values found.\n")
} 

# Plot missing values
gg_miss_var(data, show_pct = TRUE)  # Bar plot of missing values
gg_miss_upset(data)  # Upset plot for combinations of missing values
gg_miss_case(data)  # Heatmap-like visualization of missing data across cases

# Visualize the missing value pattern using a heatmap
library(visdat)
vis_miss(data)


# Step 2: Handle Outliers
# Skipped because there are no continuous variables

# Drop 'car' column and low variability column 'toCoupon_GEQ5min'
data <- data %>% select(-car, -toCoupon_GEQ5min,-temperature)


# Frequency table for each categorical variable
frequency_tables <- lapply(data, table)
print(frequency_tables)

# Step 3: Group low-frequency categories in the 'occupation' column
# Group similar occupations into broader categories for better modeling
data$occupation_class <- NA
data$occupation_class[data$occupation %in% c("Healthcare Support", "Healthcare Practitioners & Technical")] <- "Healthcare"
data$occupation_class[data$occupation %in% c("Life Physical Social Science", "Community & Social Services")] <- "Social"
data$occupation_class[data$occupation %in% c("Construction & Extraction", "Installation Maintenance & Repair", 
                                             "Building & Grounds Cleaning & Maintenance", 
                                             "Transportation & Material Moving", "Food Preparation & Serving Related")] <- "Trade Workers"
data$occupation_class[data$occupation %in% c("Sales & Related", "Management", "Office & Administrative Support",
                                             "Business & Financial", "Legal", "Education&Training&Library",
                                             "Architecture & Engineering", "Arts Design Entertainment Sports & Media",
                                             "Computer & Mathematical")] <- "White Collar"
data$occupation_class[data$occupation == "Retired"] <- "Retired"
data$occupation_class[data$occupation == "Student"] <- "Student"
data$occupation_class[data$occupation %in% c("Production Occupations", "Protective Service", 
                                             "Personal Care & Service", "Farming Fishing & Forestry")] <- "Others"
data$occupation_class[data$occupation == "Unemployed"] <- "Unemployed"
data$occupation_class <- factor(data$occupation_class)
#data$occupation <- NULL # Drop the original 'occupation' column

# Drop original 'occupation' column
data <- data %>% select(-occupation)


# Step 4: Convert categorical variables into dummy variables
# Use dummyVars from caret package
data <- dummyVars(" ~ .", data = data) %>% predict(newdata = data) %>% as.data.frame()

# Step 5: Remove highly correlated variables
# Compute correlation matrix and remove variables with correlation > 0.8
cor_matrix <- cor(data)
print(cor_matrix)  # View the correlation matrix
# to handle NA values
anyNA(cor_matrix)
cor_matrix[is.na(cor_matrix)] <- 0
cor_matrix <- cor_matrix[complete.cases(cor_matrix), complete.cases(t(cor_matrix))]
high_corr <- findCorrelation(cor_matrix, cutoff = 0.8)
data <- data[, -high_corr]

# Split data into training and test sets (70% train, 30% test)
set.seed(42)
train_index <- createDataPartition(data$Y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Split data into predictors and response for modeling
X_train <- train_data %>% select(-Y)
X_test <- test_data %>% select(-Y)
y_train <- train_data$Y
y_test <- test_data$Y

# Logistic Regression Models
# Build and evaluate logistic regression models using different selection methods
logistic_model <- glm(Y ~ ., data = train_data, family = binomial)
forward_model <- step(glm(Y ~ 1, data = train_data, family = binomial), scope = list(lower = ~1, upper = ~.), direction = "forward")
summary(forward_model)
backward_model <- step(logistic_model, direction = "backward")
summary(backward_model)
stepwise_model <- step(logistic_model, direction = "both")
summary(stepwise_model)


#####
# Predictions for All Logistic Regression Models
predict_and_evaluate <- function(model, cutoff) {
  y_pred_prob <- predict(model, as.data.frame(test_data), type = "response")
  y_pred <- ifelse(y_pred_prob > cutoff, 1, 0)
  # Ensure factor levels are aligned
  y_pred <- factor(y_pred, levels = levels(factor(y_test)))
  cm <- confusionMatrix(factor(y_pred), factor(y_test))
  val_er <- 1 - cm$overall['Accuracy']
  sensitivity <- cm$byClass['Sensitivity']
  specificity <- cm$byClass['Specificity']
  list(ValidationError = val_er, Sensitivity = sensitivity, Specificity = specificity)
  print(levels(factor(y_test)))
  print(levels(factor(y_pred)))
}

# Evaluate each model at Cutoff 0.3, 0.4, 0.5
cutoffs <- c(0.3, 0.4, 0.5)
metrics_forward <- lapply(cutoffs, function(c) predict_and_evaluate(forward_model, c))
metrics_backward <- lapply(cutoffs, function(c) predict_and_evaluate(backward_model, c))
metrics_stepwise <- lapply(cutoffs, function(c) predict_and_evaluate(stepwise_model, c))

# Print Metrics
print("Forward Selection Metrics:")
print(metrics_forward)
print("Backward Selection Metrics:")
print(metrics_backward)
print("Stepwise Selection Metrics:")
print(metrics_stepwise)
#####


# Function to Evaluate Models
#evaluate_model <- function(model, cutoff, X_test, y_test) {
 # pred_probs <- predict(model, newdata = as.data.frame(X_test), type = "response")
  #predicted_class <- ifelse(pred_probs >= cutoff, 1, 0)
  #cm <- table(Predicted = predicted_class, Actual = y_test)
  #sensitivity <- ifelse(ncol(cm) >= 2, cm[2, 2] / (cm[2, 2] + cm[2, 1]), 0)
  #specificity <- ifelse(ncol(cm) >= 2, cm[1, 1] / (cm[1, 1] + cm[1, 2]), 0)
  #val_er <- 1 - sum(diag(cm)) / sum(cm)
  #return(data.frame(Cutoff = cutoff, Sensitivity = sensitivity, Specificity = specificity, Validation_Error = val_er))
#}

#logistic_results <- lapply(c(0.3, 0.4, 0.5), function(cutoff) {
#  evaluate_model(logistic_model, cutoff, X_test, y_test)
#})
#logistic_results_df <- do.call(rbind, logistic_results)
#print("Logistic Regression Metrics:")
#print(logistic_results_df)


# KNN Model
# Ensure Y is a factor and prepare for KNN modeling
# TrainControl for cross-validation
# Train and evaluate KNN model
train_data$Y <- factor(train_data$Y, levels = c(0, 1), labels = c("Class0", "Class1"))
test_data$Y <- factor(test_data$Y, levels = c(0, 1), labels = c("Class0", "Class1"))

# TrainControl with ClassProbs
trControl <- trainControl(method = "cv", classProbs = TRUE)

# Train KNN Model
knn_model <- train(Y ~ ., data = train_data, method = "knn", tuneLength = 5, trControl = trControl)

# Predict Probabilities
knn_probs <- predict(knn_model, X_test, type = "prob")[, 2]

# Evaluate Metrics
knn_results <- lapply(c(0.3, 0.4, 0.5), function(cutoff) {
  predicted_class <- ifelse(knn_probs >= cutoff, 1, 0)
  cm <- table(Predicted = predicted_class, Actual = y_test)
  sensitivity <- ifelse(ncol(cm) >= 2, cm[2, 2] / (cm[2, 2] + cm[2, 1]), 0)
  specificity <- ifelse(ncol(cm) >= 2, cm[1, 1] / (cm[1, 1] + cm[1, 2]), 0)
  val_er <- 1 - sum(diag(cm)) / sum(cm)
  return(data.frame(Cutoff = cutoff, Sensitivity = sensitivity, Specificity = specificity, Validation_Error = val_er))
})

knn_results_df <- do.call(rbind, knn_results)
print(knn_results_df)



# Combine Results
results <- list(Logistic = logistic_results_df, KNN = knn_results_df)
results

# CART Model
# Split the dataset
set.seed(42)
train_index <- createDataPartition(data$Y, p = 0.7, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Fit CART Model
cart_model <- rpart(Y ~ ., data = train_data, method = "class", xval = 10, minsplit = 5)

# Minimum Error Tree
pfit_me <- prune(cart_model, cp = cart_model$cptable[which.min(cart_model$cptable[,"xerror"]),"CP"])
rpart.plot(
  pfit_me,
  main = "Minimum Error Tree",
  type = 2,             # Show split labels on edges of the tree
  extra = 100,          # Show only predicted class and proportion of data in each node
  fallen.leaves = TRUE, # Align leaf nodes horizontally
  shadow.col = "gray",  # Add shadow for better visualization
  box.palette = "RdYlGn", # Use a clear and contrasting color palette
  cex = 0.7,            # Reduce text size for better readability
  tweak = 1.1,          # Adjust layout spacing
  split.cex = 0.8,      # Adjust split text size
  split.box.col = "lightblue", # Background color for split text boxes
  split.border.col = "darkblue", # Border color for split boxes
  split.round = 0.4     # Add rounded corners to split boxes
)
# Best Pruned Tree
K <- 10  # Number of cross-validations
ind <- which.min(cart_model$cptable[,"xerror"])  # Find index of min xerror
se1 <- cart_model$cptable[ind, "xstd"] / sqrt(K)  # Compute 1 standard error
xer1 <- min(cart_model$cptable[,"xerror"]) + se1  # Target error
ind0 <- which.min(abs(cart_model$cptable[1:ind, "xerror"] - xer1))  # Closest xerror
pfit_bp <- prune(cart_model, cp = cart_model$cptable[ind0, "CP"])
#rpart.plot(pfit_bp, main = "Best Pruned Tree")
rpart.plot(
  pfit_bp,
  main = "Best Pruned Tree",
  type = 2,             # Show split labels on edges of the tree
  extra = 100,          # Show only the predicted class and proportion of data in each node
  fallen.leaves = TRUE, # Align leaf nodes horizontally
  shadow.col = "gray",  # Add shadow for visual depth
  box.palette = "RdYlGn", # Use a simple and non-intrusive color palette
  cex = 0.7,            # Reduce text size for better readability
  tweak = 1.1,          # Adjust layout for better spacing
  split.cex = 0.8,      # Adjust split text size
  split.box.col = "lightblue", # Background color for split text boxes
  split.border.col = "darkblue", # Border color for split boxes
  split.round = 0.4     # Add rounded corners to split boxes
)

# Evaluate Minimum Error Tree
prob_me <- predict(pfit_me, test_data, type = "prob")[, 2]
class_me <- ifelse(prob_me > 0.5, 1, 0)
cm_me <- table(Predicted = class_me, Actual = test_data$Y)
accuracy_me <- sum(diag(cm_me)) / sum(cm_me)
cat("Accuracy (Min Error Tree):", accuracy_me, "\n")

# Evaluate Best Pruned Tree
prob_bp <- predict(pfit_bp, test_data, type = "prob")[, 2]
class_bp <- ifelse(prob_bp > 0.5, 1, 0)
cm_bp <- table(Predicted = class_bp, Actual = test_data$Y)
accuracy_bp <- sum(diag(cm_bp)) / sum(cm_bp)
cat("Accuracy (Best Pruned Tree):", accuracy_bp, "\n")

# Alternative Cutoff Evaluation
evaluate_cart <- function(probs, cutoff, actual) {
  pred <- ifelse(probs > cutoff, 1, 0)
  cm <- table(Predicted = pred, Actual = actual)
  sensitivity <- ifelse(ncol(cm) >= 2, cm[2, 2] / (cm[2, 2] + cm[2, 1]), 0)
  specificity <- ifelse(ncol(cm) >= 2, cm[1, 1] / (cm[1, 1] + cm[1, 2]), 0)
  val_error <- 1 - sum(diag(cm)) / sum(cm)
  return(data.frame(Cutoff = cutoff, Sensitivity = sensitivity, Specificity = specificity, Validation_Error = val_error))
}

# Evaluate at Different Cutoffs for Best Pruned Tree
cutoffs <- c(0.3, 0.4, 0.5)
cart_results <- do.call(rbind, lapply(cutoffs, function(c) evaluate_cart(prob_bp, c, test_data$Y)))

cat("CART Results (Best Pruned Tree):\n")
print(cart_results)


# Combine Results
results <- list(Logistic = logistic_results_df, KNN = knn_results_df, CART = cart_results)
results

# ROC Curves and AUC
plot_roc_curve <- function(pred_probs, title, y_test) {
  roc_curve <- roc(y_test, pred_probs)
  plot(roc_curve, main = title, col = "blue", lwd = 2)
  legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "blue", lwd = 2)
}

plot_roc_curve(knn_probs, "ROC Curve for KNN", y_test)
plot_roc_curve(prob_me, "ROC Curve for CART", y_test)
plot_roc_curve(prob_bp, "ROC Curve for Pruned CART", y_test)