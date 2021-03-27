#Projeto Aplicado
#Pós Graduação - Ciência de Dados

# Simular predição 

rm(list = ls())

library('readxl')

#Cria o data frame
dados <- read_excel('~/Downloads/projeto-aplicado-simulacao-2018.xlsx')

View(dados)

#Ajusta um modelo de regressao linear multipla
dev.off()
modelo <- lm(QTD_VENDA ~ PERIODO+SEGMENTO_AREA+AREA+EMPRESA+`IPC-BR`+`IGP-M`+IPCA, data = dados)

summary(modelo) 
 #dev.off()

#Iremos criar um data frame sem a variavel resposta vendas do cafe, pois ela sera estimada pela equacao de regressao que ajustamos
dados_para_predicao <- read_excel('~/Downloads/projeto-aplicado-simulacao-predicao.xlsx')

#Estima a variavel resposta pra cada observacao do novo data frame
predicoes <- predict(modelo, newdata = dados_para_predicao)

View (data.frame(dados_para_predicao, predicoes))

