SELECT PERIODO, 
VARIAVEL, 
SUM(QTD_VENDA), 
SUM(VALOR_VENDA),  
SUM(PRECO_MEDIO)  
FROM (SELECT T.DS_ANO_MES PERIODO 
,'GEN' VARIAVEL 
,V.QT_ITEM_VENDA QTD_VENDA 
,V.VL_BRUTO_VENDA VALOR_VENDA 
,0 PRECO_MEDIO 
FROM SCH_ODS.FATO_VENDA V, SCH_ODS.DIM_PRODUTO P, SCH_ODS.DIM_TEMPO T 
WHERE V.SK_TEMPO BETWEEN 20150101 AND 20193112 
AND P.SK_PRODUTO = V.SK_PRODUTO 
AND T.SK_TEMPO = V.SK_TEMPO 
AND P.CD_TIPO_PUBLICACAO = '1') 
GROUP BY PERIODO, VARIAVEL 
ORDER BY 1