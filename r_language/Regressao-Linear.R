#Projeto Aplicado
#Pós Graduação - Ciência de Dados

#Regressão Linear - análise de variáveis

#Limpa memória do R
rm(list = ls())

library('readxl')

#Cria o data frame
dados <- read_excel('~/Downloads/projeto-aplicado-simulacao-2018.xlsx')
#dados = read.table('/home/rafael/Downloads/projeto-aplicado-simulacao-2018.xlsx - Planilha3.tsv', header=T)
#attach(dados)

View(dados)

#Ajusta um modelo de regressao linear multipla
dev.off()
modelo <- lm(QTD_VENDA ~ ., data = dados)

#Visualiza resumo do ajuste do modelo
summary(modelo)

#Analisa melhores variaveis - Stepway
modelo <- step(modelo, direction = 'both')
warnings()

#Resultado Stepwise removeu variaveis
summary(modelo) 
