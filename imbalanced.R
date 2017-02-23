# imbalanced.R
# Implementation and evaluation of imbalanced classification techniques 
# Programming code courtesy by Sarah Vluymans, Sarah.Vluymans@UGent.be


########################
#### SCUBUS DATASET ####
########################

## load the subclus dataset
subclus <- read.table("subclus.txt", sep=",")
colnames(subclus) <- c("Att1", "Att2", "Class")

# determine the imbalance ratio
unique(subclus$Class)
nClass0 <- sum(subclus$Class == 0)
nClass1 <- sum(subclus$Class == 1)
IR <- nClass1 / nClass0
IR

# visualize the data distribution
plot(subclus$Att1, subclus$Att2)
points(subclus[subclus$Class==0,1],subclus[subclus$Class==0,2],col="red")
points(subclus[subclus$Class==1,1],subclus[subclus$Class==1,2],col="blue")  

# Set up the dataset for 5 fold cross validation.
# Make sure to respect the class imbalance in the folds.
pos <- (1:dim(subclus)[1])[subclus$Class==0]
neg <- (1:dim(subclus)[1])[subclus$Class==1]

CVperm_pos <- matrix(sample(pos,length(pos)), ncol=5, byrow=T)
CVperm_neg <- matrix(sample(neg,length(neg)), ncol=5, byrow=T)

CVperm <- rbind(CVperm_pos, CVperm_neg)

# Base performance of 3NN
library(class)
knn.pred = NULL
for( i in 1:5){
  predictions <- knn(subclus[-CVperm[,i], -3], subclus[CVperm[,i], -3], subclus[-CVperm[,i], 3], k = 3)
  knn.pred <- c(knn.pred, predictions)
}
acc <- sum((subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) 
           | (subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean <- sqrt(tpr * tnr)


# 1. ROS (Random Oversampling)
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices)
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, train[duplicate[j],])
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS <- sqrt(tpr.ROS * tnr.ROS)

# 2. RUS (Random Undersampling)
knn.pred = NULL
for( i in 1:5){
  
  train <- subclus[-CVperm[,i], -3]
  classes.train <- subclus[-CVperm[,i], 3] 
  test  <- subclus[CVperm[,i], -3]
  
  # randomly undersample the minority class (class 1)
  majority.indices <- (1:dim(train)[1])[classes.train == 1]
  to.remove <- 2* length(majority.indices) - dim(train)[1]
  remove <- sample(majority.indices, to.remove, replace = F)
  train <- train[-remove,] 
  classes.train <- classes.train[-remove]
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS <- sum(subclus$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS <- sqrt(tpr.RUS * tnr.RUS)

# Visualization (RUS on the full dataset)
subclus.RUS <- subclus
majority.indices <- (1:dim(subclus.RUS)[1])[subclus.RUS$Class == 1]
to.remove <- 2 * length(majority.indices) - dim(subclus.RUS)[1]
remove <- sample(majority.indices, to.remove, replace = F)
subclus.RUS <- subclus.RUS[-remove,] 

plot(subclus.RUS$Att1, subclus.RUS$Att2)
points(subclus.RUS[subclus.RUS$Class==0,1],subclus.RUS[subclus.RUS$Class==0,2],col="red")
points(subclus.RUS[subclus.RUS$Class==1,1],subclus.RUS[subclus.RUS$Class==1,2],col="blue") 



########################
#### CIRCLE DATASET ####
########################

## Cargamos el fichero circle.txt
circle <- read.table("circle.txt", sep=",")
colnames(circle) <- c("Att1", "Att2", "Class")

# Calculamos IR = ratio de imbalanceo
unique(circle$Class)
nClass0 <- sum(circle$Class == 0)
nClass1 <- sum(circle$Class == 1)
IR <- nClass1 / nClass0
IR

# Visualizamos el dataset
plot(circle$Att1, circle$Att2)
points(circle[circle$Class==0,1],circle[circle$Class==0,2],col="red")
points(circle[circle$Class==1,1],circle[circle$Class==1,2],col="blue")  

# Separamos en 5fcv manteniendo el ratio de imbalanceo
pos <- (1:dim(circle)[1])[circle$Class==0]
neg <- (1:dim(circle)[1])[circle$Class==1]

CVperm_pos <- matrix(sample(pos,length(pos)), ncol=5, byrow=T)
CVperm_neg <- matrix(sample(neg,length(neg)), ncol=5, byrow=T)

CVperm <- rbind(CVperm_pos, CVperm_neg)

# Aplicamos KNN con K = 3
library(class)
knn.pred = NULL
for( i in 1:5){
  predictions <- knn(circle[-CVperm[,i], -3], circle[CVperm[,i], -3], circle[-CVperm[,i], 3], k = 3)
  knn.pred <- c(knn.pred, predictions)
}
acc <- sum((circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) 
           | (circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2)) / (nClass0 + nClass1)
tpr <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean <- sqrt(tpr * tnr)


# 1. ROS (Random Oversampling)
knn.pred = NULL
for( i in 1:5){
  train <- circle[-CVperm[,i], -3]
  classes.train <- circle[-CVperm[,i], 3] 
  test  <- circle[CVperm[,i], -3]
  
  # randomly oversample the minority class (class 0)
  minority.indices <- (1:dim(train)[1])[classes.train == 0]
  to.add <- dim(train)[1] - 2 * length(minority.indices)
  duplicate <- sample(minority.indices, to.add, replace = T)
  for( j in 1:length(duplicate)){
    train <- rbind(train, train[duplicate[j],])
    classes.train <- c(classes.train, 0)
  }  
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.ROS <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.ROS <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.ROS <- sqrt(tpr.ROS * tnr.ROS)


# 2. RUS (Random Undersampling)
knn.pred = NULL
for( i in 1:5){
  
  train <- circle[-CVperm[,i], -3]
  classes.train <- circle[-CVperm[,i], 3] 
  test  <- circle[CVperm[,i], -3]
  
  # randomly undersample the minority class (class 1)
  majority.indices <- (1:dim(train)[1])[classes.train == 1]
  to.remove <- 2* length(majority.indices) - dim(train)[1]
  remove <- sample(majority.indices, to.remove, replace = F)
  train <- train[-remove,] 
  classes.train <- classes.train[-remove]
  
  # use the modified training set to make predictions
  predictions <-  knn(train, test, classes.train, k = 3)
  knn.pred <- c(knn.pred, predictions)
}
tpr.RUS <- sum(circle$Class[as.vector(CVperm)] == 0 & knn.pred == 1) / nClass0
tnr.RUS <- sum(circle$Class[as.vector(CVperm)] == 1 & knn.pred == 2) / nClass1
gmean.RUS <- sqrt(tpr.RUS * tnr.RUS)

# Visualization (RUS on the full dataset)
circle.RUS <- circle
majority.indices <- (1:dim(circle.RUS)[1])[circle.RUS$Class == 1]
to.remove <- 2 * length(majority.indices) - dim(circle.RUS)[1]
remove <- sample(majority.indices, to.remove, replace = F)
circle.RUS <- circle.RUS[-remove,] 

plot(circle.RUS$Att1, circle.RUS$Att2)
points(circle.RUS[circle.RUS$Class==0,1],circle.RUS[circle.RUS$Class==0,2],col="red")
points(circle.RUS[circle.RUS$Class==1,1],circle.RUS[circle.RUS$Class==1,2],col="blue") 



###############
#### SMOTE ####
###############

# 2.4.1 Distance function
distance <- function(i, j, data){
  sum <- 0
  for(f in 1:dim(data)[2]){
    if(is.factor(data[,f])){ # nominal feature
      if(data[i,f] != data[j,f]){
        sum <- sum + 1
      }
    } else {
      sum <- sum + (data[i,f] - data[j,f]) * (data[i,f] - data[j,f])
    }
  }
  sum <- sqrt(sum)
  return(sum)
}


# 2.4.2 getNeigbors FUNCTION

getNeighbors <- function(x, minority.instances, train)
{
  distance.vector = sapply(minority.instances, distance, x, train)
  distance.and.index = cbind(distance.vector, minority.instances)
  distance.and.index = distance.and.index[order(distance.vector),]
  return (distance.and.index[2:6,2])
}

##Para testear
#getNeighbors(pos[1],pos, subclus)

# 2.4.3 syntheticInstance
train = circle
minority.instances = pos
x = pos[1]


syntheticInstance <- function(x, minority.instances, train)
{
  ## Calculamos los 5 primeros vecinos
  neighbor.index.list = getNeighbors(x,minority.instances, train)
  ## Extraemos un vecino aleatorio
  neighbor.index = neighbor.index.list[round(runif(1,1,length(neighbor.index.list)))]
  ##Obtenemos un valor entre 0 y 1 que indicara el grado
  ##de semejanza a un dato u otro (0 igual a x, 1 igual a vecino)
  similarity = runif(1)
  ##calculo del punto x (valor entre los datos usando similarity)
  distan = c(train[x,1]-train[neighbor.index,1],train[x,2]-train[neighbor.index,2]);
  new.instance = train[x,-3] + similarity * distan;
  new.instance = cbind (new.instance, train[x,3])
  ##calculo del punto y a traves de la funcion de la recta (bidimensional)
  return (new.instance)
}


##########################
#### borderline-SMOTE ####
##########################

#uses distance function of normal SMOTE

#get K neighbors from minority instances
getKNeighbors <- function(x, k, minority.instances, train)
{
  distance.vector = sapply(minority.instances, distance, x, train)
  distance.and.index = cbind(distance.vector, minority.instances)
  distance.and.index = distance.and.index[order(distance.vector),]
  if (k > length(distance.and.index))
    return (distance.and.index[-1,2])
  else
    return (distance.and.index[2:(k+1),2])
}

##Get M neighbors(index) of x
getNeighborsBorderline <- function(x, m, data)
{
  distance.vector = sapply(x, distance,1:dim(data)[1], data)
  distance.and.index = cbind(distance.vector, 1:dim(distance.vector)[1])
  distance.and.index = distance.and.index[order(distance.vector),]
  return (distance.and.index[2:(m+1),2])
}
#getNeighborsBorderline(1,5, train)

# M = number of neighbors
# K = number of possitive neighbors
# s = number of sythetic instances for each DANGER positive instance

syntheticInstancesBorderline <- function(m, k, s, positive.instances, data)
{
  # search DANGER positive instances
  danger = NULL
  for (i in positive.instances)
  {
    # Calculate its M nearest neighbors
    neighbors.index.m = getNeighborsBorderline(i, m, data)
    # Check if its in DANGER and who not (M' is number of negative neighbors)
    mPos = length(intersect(neighbors.index.m, positive.instances))
    mNeg = length(neighbors.index.m) - mPos
    # M' = M (noise)
    # M/2 <= M' < M (DANGER)
    # Others M' < M/2 (safe)
    if (mNeg < m && mNeg > (m/2))
    {
      mNeg
      danger = cbind(danger, i)
    }
    
  }
  # for each DANGER instance
  syntheticInstances = NULL
  for (i in danger)
  {
    # calculate its k nearest neighbors of Positive instances
    # and select a random S neighbors (S must be in 1:k)
    neighbors.index.s = sample(getKNeighbors(i, k, positive.instances, data), s)
    #generate synthetic instances from i to all neighbors.index.s
    for (j in neighbors.index.s)
    {
      similarity = runif(1)
      ##calculo del punto x (valor entre los datos usando similarity)
      distan = c(data[x,1]-data[j,1],data[x,2]-data[j,2]);
      new.instance = data[x,-3] + similarity * distan;
      new.instance = cbind (new.instance, data[x,3])
      colnames(new.instance) = colnames(data)
      syntheticInstances = rbind(new.instance, syntheticInstances)
    }
  }
  return (syntheticInstances);
}

# test call
#syntheticInstances = syntheticInstancesBorderline(50, 7, 4, pos, circle)


################################
## TESTING UNBALANCED LIBRARY ##
################################

library(unbalanced)

#prepare data
subclus <- read.table("subclus.txt", sep=",")
colnames(subclus) <- c("Att1", "Att2", "Class")

subclus$Class = as.factor((subclus$Class - 1) * (-1))
subclusPos <- (1:dim(subclus)[1])[subclus$Class==1]
subclusNeg <- (1:dim(subclus)[1])[subclus$Class==0]

circle <- read.table("circle.txt", sep=",")
colnames(circle) <- c("Att1", "Att2", "Class")

circle$Class = as.factor((circle$Class - 1) * (-1))
circlePos <- (1:dim(circle)[1])[circle$Class==1]
circleNeg <- (1:dim(circle)[1])[circle$Class==0]


#####################
## SMOTE + TomekLinks

## subclus
#SMOTE
n<-ncol(subclus)
output<-subclus$Class
input<-subclus[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
balancedData<-cbind(data$X,data$Y)
#TomekLinks
n<-ncol(balancedData)
output<-balancedData$Class
input<-balancedData[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubTomek", percOver=300, percUnder=150, verbose=TRUE)
subclusBalancedData<-cbind(data$X,data$Y)

## circle
#SMOTE
n<-ncol(circle)
output<-circle$Class
input<-circle[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
balancedData<-cbind(data$X,data$Y)
#TomekLinks
n<-ncol(balancedData)
output<-balancedData$Class
input<-balancedData[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubTomek", percOver=300, percUnder=150, verbose=TRUE)
circleBalancedData<-cbind(data$X,data$Y)

## comparison


##############
## SMOTE + ENN

## subclus
#SMOTE
n<-ncol(subclus)
output<-subclus$Class
input<-subclus[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
balancedData<-cbind(data$X,data$Y)
#ENN
n<-ncol(balancedData)
output<-balancedData$Class
input<-balancedData[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubENN", percOver=300, percUnder=150, verbose=TRUE)
subclusBalancedData<-cbind(data$X,data$Y)

## circle
#SMOTE
n<-ncol(circle)
output<-circle$Class
input<-circle[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubSMOTE", percOver=300, percUnder=150, verbose=TRUE)
balancedData<-cbind(data$X,data$Y)
#ENN
n<-ncol(balancedData)
output<-balancedData$Class
input<-balancedData[ ,-n]
data<-ubBalance(X= input, Y=output, type="ubENN", percOver=300, percUnder=150, verbose=TRUE)
circleBalancedData<-cbind(data$X,data$Y)


#comparison

