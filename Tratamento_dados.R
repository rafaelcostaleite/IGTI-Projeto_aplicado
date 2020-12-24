rm(list = ls())

library(dplyr) 
library(caret)

#Cria o data frame
dados = read.table('/home/rafael/Downloads/projeto-aplicado-simulacao-2018.csv', header=TRUE, sep=",")
attach(dados)

#Estatistica Descritiva - Grupo de Venda
summary(dados)

glimpse(dados)

#dados$QTD_NOR = log(dados$QTD_VENDA)

proc1 = preProcess(dados[2:14], method=c("center","scale"))
norm1 = predict(proc1,dados[2:14])

summary(norm1)

glimpse(norm1)

#Ajusta um modelo de regressao linear multipla
dev.off()
modelo <- lm(QTD_VENDA ~ ., data = norm1)

#Visualiza resumo do ajuste do modelo
summary(modelo)

#Analisa melhores variaveis - Stepway
modelo <- step(modelo, direction = 'both')
warnings()

#Resultado Stepwise removeu variaveis
summary(modelo) 

#Ajusta um modelo de regressao linear multipla
dev.off()
modelo <- lm(QTD_VENDA ~ SEGMENTO_AREA+AREA+EMPRESA+INPC+IPCA, data = norm1)

summary(modelo) 
#dev.off()

#Iremos criar um data frame sem a variavel resposta vendas do cafe, pois ela sera estimada pela equacao de regressao que ajustamos
dados_para_predicao <- read_excel('~/Downloads/projeto-aplicado-simulacao-predicao.xlsx')

#Estima a variavel resposta pra cada observacao do novo data frame
predicoes <- predict(modelo, newdata = dados_para_predicao)

View (data.frame(dados_para_predicao, predicoes))
