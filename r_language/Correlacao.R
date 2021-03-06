#Projeto Aplicado
#Pós Graduação - Ciência de Dados

#Regressão Linear - análise de variáveis
library('readxl')
library(corrplot)

#Limpa memória do R
rm(list = ls())

#Cria o data frame
dados <- read_excel('~/Downloads/projeto-aplicado-correlacao.xlsx')
dados <- as.data.frame (dados[,-1], row.names = dados[,1])

View(dados)

matrizcorrelacao <- cor(dados)

corrplot(matrizcorrelacao, method="number")

# mat : is a matrix of data
# ... : further arguments to pass to the native R cor.test function
cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}
# matrix of the p-value of the correlation
p.mat <- cor.mtest(dados)

head(p.mat[, 1:5])
