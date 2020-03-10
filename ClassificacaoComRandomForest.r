### COLETA DOS DADOS E FEATURE ENGINEERING ###

source("ClassTools.R")
Credit <- read.csv("GermanCreditCard.csv") 
metaFrame <- data.frame(colNames, isOrdered, I(factOrder))
Credit <- fact.set(Credit, metaFrame)


# Balanceamento do numero de casos positivos e negativos
Credit <- equ.Frame(Credit, 2) 

View(Credit)

# Transformando variávies numericas em variaveis categoricas (quantization)
toFactors <- c("Duration", "CreditAmount", "Age")
maxVals <- c(100, 1000000, 100)
facNames <- unlist(lapply(toFactors, function(x) paste(x, "_f", sep = "")))
Credit[, facNames] <- Map(function(x, y) quantize.num(Credit[, x], maxval = y),
                          toFactors, maxVals)



#### ANÁLISE EXPLORATÓRIA DE DADOS ####

# Plots usando o ggplot2
library(ggplot2)
lapply(colNames2, function(x) {
  if(is.factor(Credit[, x])) {
    ggplot(Credit, aes_string(x)) +
      geom_bar() +
      facet_grid(. ~ CreditStatus) +
      ggtitle(paste("Total de Crédito Bom/Ruim por ", x))
  }
})


# Plots CreditStatus vs CheckingAccStat
lapply(colNames2, function(x){
  if(is.factor(Credit[, x]) & x != "CheckingAcctStat") {
    ggplot(Credit, aes(CheckingAcctStat)) +
      geom_bar() +
      facet_grid(paste(x, " ~ CreditStatus")) +
      ggtitle(paste("Total de Crédito Bom/Ruim CheckingAccStat e ", x))
  }
})



### FEATURE SELECTION UTILIZANDO RANDOM FOREST ###

library(randomForest)

modelo_rf <- randomForest(CreditStatus ~ .
                          - Duration
                          - Age
                          - CreditAmount
                          - ForeignWorker
                          - NumberDependents
                          - Telephone
                          - ExistingCreditsAtBank
                          - PresentResidenceTime
                          - Job
                          - Housing
                          - SexAndStatus
                          - InstallmentRatePecnt
                          - OtherDetorsGuarantors
                          - Age_f
                          - OtherInstalments,
                          data = Credit,
                          ntree = 100, nodesize = 10, importance = T)


varImpPlot(modelo_rf, main = "Variaveis Mais Significativas")



### CRIANDO O MODELO PREDITIVO NO R ###

# Criar um modelo de classificação baseado no RandomForest
library(randomForest)

# Cross Tabulation - Tabela Cruzada
?table
table(Credit$CreditStatus)

# Funçao para gerar dados de treino e dados de teste
splitData <- function(dataFrame, seed = NULL) {
  if(!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataFrame)
  trainIndex <- sample(index, trunc(length(index)/2))
  trainSet <- dataFrame[trainIndex, ]
  testSet <- dataFrame[-trainIndex, ]
  list(trainSet = trainSet, testSet = testSet)  
}

# Gerando dados de treino e de teste
splits <- splitData(Credit, seed = 808)

# Separando os dados
dados_teste <- splits$testSet
dados_treino <- splits$trainSet

# Verificando o numero de linhas 
nrow(dados_treino)
nrow(dados_teste)

# Construindo o modelo de Random Forest
modelo_rf <- randomForest( CreditStatus ~ CheckingAcctStat
                           + Duration_f
                           + Purpose
                           + CreditHistory
                           + SavingsBonds
                           + Employment
                           + CreditAmount_f,
                           data = dados_treino,
                           ntree = 100,
                           nodesize = 10
)

# Imprimindo o resultado
print(modelo_rf)



###### GERANDO OS SCORES DO MODELO #######

# Fazendo Previsões

# Previsoes com um modelo de classificação baseado em Random Forest
require(randomForest)

# Gerando as previsoes nos dados de teste
previsoes <- data.frame(observado = dados_teste$CreditStatus,
                        previsto = predict(modelo_rf, newdata = dados_teste))

# Visualizando os resultados
View(previsoes) # Tabela com os dados previstos
View(dados_teste) # Tabela com os dados de teste



##### AVALIANDO O MODELO ######


# FORMULAS PARA AS METRICAS DE AVALIAÇÃO

Accuracy <- function(x) {
  (x[1,1] + x[2,2]) / (x[1,1] + x[1,2] + x[2,1] + x[2,2]) 
}


Recall <- function(x) {
  x[1,1] / (x[1,1] + x[1,2])
}

Precision <- function(x) {
  x[1,1] / (x[1,1] + x[2,1])
} 

W_Accuracy <- function(x) {
  (x[1,1] + x[2,2]) / (x[1,1] + 5 * x[1,2] + x[2,1] + x[2,2])
}

  
F1 <- function(x) {
  (2 * x[1,1]) /  (2 * x[1,1] + x[1,2] + x[2,1])
}
  

# Criando a Confusion Matrix Manualmente.
confMat <- matrix(unlist(Map(function(x, y) {
                  sum(ifelse(previsoes[ ,1] == x & previsoes[, 2] == y, 1, 0))},
                  c(2, 1, 2, 1), c(2, 2, 1, 1))), nrow = 2)


# Criando um dataframe com as estatisticas dos testes
df_mat <- data.frame(Category = c("Credito Ruim", "Credito Bom"),
                     Classificado_como_ruim = c(confMat[1,1], confMat[2,1]),
                     Classificado_como_bom = c(confMat[1,2], confMat[2,2]),
                     Accuracy_Recall = c(Accuracy(confMat), Recall(confMat)),
                     Precision_WAcc = c(Precision(confMat), W_Accuracy(confMat)))

print(df_mat)



# Gerando Uma Curva ROC em R
install.packages("ROCR")
library(ROCR)

# Gerando as classes de dados
class1 <- predict(modelo_rf, newdata = dados_teste, type = "prob")
class2 <- dados_teste$CreditStatus
  
# Gerando a curva ROC
?prediction
?performance  
pred <- prediction(class1[, 2], class2)
perf <- performance(pred, "tpr", "fpr")
plot(perf, col = rainbow(10))


# Gerando Confusion Matrix com o Caret
library(caret)
?confusionMatrix
confusionMatrix(previsoes$observado, previsoes$previsto)



### OTIMIZANDO O MODELO ### 

# Modelo randomForest ponderado
# O pacote C50 permite que você de peso aos erros, construindo 
# assim um resultado ponderado
install.packages("C50")
install.packages("partykit")
library(C50)

# Criando uma Cost Function
Cost_func <- matrix(c(0, 1.5, 1, 0), nrow = 2, dimnames = list(c("1", "2"), c("1", "2")))

# Criando o Modelo
?randomForest
?C5.0

# Cria o modelo
modelo_rf_v2 <- C5.0(CreditStatus ~ CheckingAcctStat
                     + Purpose
                     + CreditHistory
                     + SavingsBonds
                     + Employment,
                     data = dados_treino,
                     trials = 100,
                     cost = Cost_func)

print(modelo_rf_v2)

# Dataframe com valores observados e previstos
previsoes_v2 <- data.frame(observado = dados_teste$CreditStatus,
                           previsto = predict(object = modelo_rf_v2, newdata = dados_teste))