# %%
#instalação de pacotes
#!pip3 install pyramid-arima
#!pip install pmdarima
#!pip install mysql-connector-python
#!pip install keras

#Para efetuar a conversão de notebook para script
#ipynb-py-convert python-time-series-cron.ipynb python-time-series-cron.py

# %%
"""
# Seleciona Produtos
"""

# %%
import numpy as np
import pandas as pd
import mysql.connector

#Conecta com banco MySQL
try:
    connection = mysql.connector.connect(host='localhost',
                                         database='projeto',
                                         user='root',
                                         password='12345678')
    
    sql_select_Query = " select sk_produto \
                        from dim_produto as pd \
                        where pd.sk_produto > 0 "
    
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    
    #carrega dados
    records = cursor.fetchall()

    data_sql = pd.DataFrame(records)

except mysql.connector.Error as e:
    print("Error reading data from MySQL table", e)
finally:
    if connection.is_connected():
        connection.close()
        cursor.close()
        print("MySQL connection is closed")
        
df_prod = data_sql.rename(columns={0: "sku"})
        
df_prod.info()

# %%
def movimento_real(produto):
    
    #Conecta com banco MySQL
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='projeto',
                                             user='root',
                                             password='12345678')

        sql_select_Query = " select ds_data_sem_hora \
                            ,isbn_produto \
                            ,qtd_venda  \
                            from fato_venda as ft \
                            , dim_tempo as tp \
                            , dim_produto as pd \
                            where tp.sk_tempo = ft.sk_tempo \
                            and pd.sk_produto = ft.sk_protudo \
                            and pd.sk_produto = " + produto 

        cursor = connection.cursor()
        cursor.execute(sql_select_Query)

        #carrega dados
        records = cursor.fetchall()

        data_sql = pd.DataFrame(records)

    except mysql.connector.Error as e:
        print("Error reading data from MySQL table", e)
    finally:
        if connection.is_connected():
            connection.close()
            cursor.close() 
            print("MySQL connection is closed")

    if len(data_sql)>0:
        
        print("data_sql>0")
        
        #ajusta dados para a predição
        df_mov = data_sql.copy()

        #adiciona nome nas colunas
        df_mov = df_mov.rename(columns={0: "date",1:"isbn", 2: "value"})

        #transforma para datetime
        df_mov["date"] = pd.to_datetime(df_mov["date"])

        #ordena
        df_mov = df_mov.sort_values('date')

        #soma quantidade em datas iguais
        df_mov = df_mov.groupby('date')['value'].sum().reset_index()

        #transforma a data em indice
        df_mov = df_mov.set_index('date')

        #agrupa na soma do mês
        df_mov = df_mov['value'].resample('MS').sum()
    else: 
        
        print("data_sql=0")
        
        df_mov = []
    
    return df_mov


# %%
import pmdarima as pm

def predicao_arima(df_arima,n_periods):
    
    predicao = []
    
    #Separa dados
    train = df_arima.iloc[:len(df_arima)-12]
    test = df_arima.iloc[len(df_arima)-12:]

    #Fit modelo auto-arima
    fitSArima = pm.auto_arima(df_arima, start_p=1, start_q=1,max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True, d=None, D=1, trace=False,
                             error_action='ignore',suppress_warnings=True, stepwise=True)

    fitSArima.fit(train)
    
    #Forecast auto-arima
    index_pred = pd.date_range(test.index[0], periods = n_periods, freq='MS')

    predicao, confint = fitSArima.predict(n_periods=n_periods, return_conf_int=True)

    #Cria dataseries
    predicao = pd.DataFrame(predicao,index = index_pred,columns=['SARIMA'])
    
    return predicao

# %%
from fbprophet import Prophet

def predicao_prophet(df_prophet,n_periods):
    
    predicao = []
    
    #Quantida de períodos futuros
    fut_per = int(n_periods/2)
    
    #Ajust dados
    df_prophet = df_prophet.to_frame()

    #Reinicia indice
    df_prophet = df_prophet.reset_index()

    #adiciona nome nas colunas
    df_prophet = df_prophet.rename(columns={"date":"ds","value":"y"})
    
    #Fit modelo Prophet
    fitProphet = Prophet()

    fitProphet.fit(df_prophet)
    
    #Forecast Prophet
    future = fitProphet.make_future_dataframe(periods=fut_per, freq='MS')

    predProphetfull = fitProphet.predict(future)

    predicao = predProphetfull[['ds','yhat']]

    predicao = predicao.set_index('ds')
    predicao = predicao.rename(columns={"yhat": "PROPHET"})

    predicao = predicao.iloc[-24:]

    return predicao

# %%
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def predicao_lstm(df_lstm,n_periods):
    
    predicao = []

    #Copia dos dados tratados
    df_lstm = df_lstm.to_frame()

    #Cria data frame train e teste
    train = df_lstm.iloc[:len(df_lstm)-12]
    test = df_lstm.iloc[len(df_lstm)-12:]
    
    scaler = MinMaxScaler()
    
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    
    n_input = n_periods
    n_features = 1
    
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=2)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mse')

    #lstm_model.summary()
    
    nepochs=50
    lstm_model.fit_generator(generator,epochs=nepochs)
    
    lstm_predictions_scaled = list()

    batch = scaled_train[-n_input:]
    current_batch = batch.reshape((1, n_input, n_features))

    for i in range(n_input):   
        lstm_pred = lstm_model.predict(current_batch)[0]
        lstm_predictions_scaled.append(lstm_pred) 
        current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
        
        predicao = scaler.inverse_transform(lstm_predictions_scaled)
    
    return predicao

# %%
def grava_fato_predicao(produto_proc,df_export):
    
    df_export["sk_tempo"] = df_export.index.strftime('%Y%m%d').astype(int)
    df_export["sk_protudo"] = int(produto_proc)

    df_export = df_export.rename(columns={"value":"qtd_venda","Ac_venda":"acm_venda"})
    df_export = df_export.rename(columns={"PROPHET":"qtd_prophet","Ac_PROPHET":"acm_prophet"})
    df_export = df_export.rename(columns={"LSTM":"qtd_lstm","Ac_LSTM":"acm_lstm"})
    df_export = df_export.rename(columns={"SARIMA":"qtd_sarima","Ac_SArima":"acm_sarima"})
    df_export = df_export.replace(np.nan,0)
    
    try:
        
        print("Atualiza fato predição")
        
        # Connect to the database
        connection = mysql.connector.connect(host='localhost',
                                             database='projeto',
                                             user='root',
                                             password='12345678')

        cursor = connection.cursor()

        # Create a new record
        query = " DELETE FROM projeto.fato_predicao WHERE sk_protudo = " + str(int(produto_proc )).strip()

        cursor.execute(query)

        # connection is not autocommit by default. So we must commit to save our changes.
        connection.commit()

        # Create a new record
        #query = """INSERT INTO projeto.fato_predicao (sk_tempo, sk_protudo, qtd_predicao, qtd_acumula) VALUES(%s, %s, %s, %s) ON DUPLICATE KEY UPDATE qtd_predicao=%s, qtd_acumula=%s"""
        #cursor.execute(query, (df_export.sk_tempo[i], df_export.sk_protudo[i], df_export.PROPHET[i], df_export.Ac_PROPHET[i], df_export.PROPHET[i], df_export.Ac_PROPHET[i]))

        query = """INSERT INTO projeto.fato_predicao (sk_tempo, sk_protudo, no_modelo, qtd_predicao, qtd_acumula) VALUES(%s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE qtd_predicao=%s, qtd_acumula=%s"""

        for index, row in df_export.iterrows():
            #Grava dados da Venda Realizada
            cModelo = 'Venda'
            nQtdInsert = row['qtd_venda'].item() 
            nAcmInsert = row['acm_venda'].item()

            cursor.execute(query, (row['sk_tempo'].item(), row['sk_protudo'].item(), cModelo, nQtdInsert, nAcmInsert, nQtdInsert, nAcmInsert))       

            #Grava dados PROPHET
            cModelo = 'Prophet'
            nQtdInsert = row['qtd_prophet'].item() 
            nAcmInsert = row['acm_prophet'].item()

            cursor.execute(query, (row['sk_tempo'].item(), row['sk_protudo'].item(), cModelo, nQtdInsert, nAcmInsert, nQtdInsert, nAcmInsert))       

            #Grava dados SArima        
            cModelo = 'SArima'
            nQtdInsert = row['qtd_sarima'].item() 
            nAcmInsert = row['acm_sarima'].item()

            cursor.execute(query, (row['sk_tempo'].item(), row['sk_protudo'].item(), cModelo, nQtdInsert, nAcmInsert, nQtdInsert, nAcmInsert))       

            #Grava dados LSTM        
            cModelo = 'LSTM'
            nQtdInsert = row['qtd_lstm'].item() 
            nAcmInsert = row['acm_lstm'].item()

            cursor.execute(query, (row['sk_tempo'].item(), row['sk_protudo'].item(), cModelo, nQtdInsert, nAcmInsert, nQtdInsert, nAcmInsert))       

        # connection is not autocommit by default. So we must commit to save our changes.
        connection.commit()

        # Execute query
        sql = "SELECT * FROM projeto.fato_predicao WHERE sk_protudo = " + str(int(produto_proc )).strip()

        cursor.execute(sql)

        # Fetch all the records
        result = cursor.fetchall()

        #for i in result:
            #print(i)

    finally:
        
        # close the database connection using close() method.
        connection.close()


# %%
#Numero de períodos para predição
n_periods = 24

for ind in df_prod.index:
    
    print("produto " + str(int(df_prod['sku'][ind])) )
          
    #Carrega movimentos do produto
    df1 = movimento_real(str(int(df_prod['sku'][ind])).strip())
    
    if len(df1)>0:
        
        #Trata outliers superiores 
        #df = df.where(((df-df.mean()).abs() < 3*df.std()),3*df.std())
        df1 = df1.where(df1<df1.quantile(0.95),df1.quantile(0.95))

        #Calcula Predição - Modelo SArima
        predSArima = predicao_arima(df1,n_periods)

        predProphet = predicao_prophet(df1,n_periods)
        
        predLSTM = predicao_lstm(df1,n_periods)
        
        #Concatena predições
        test = df1.iloc[len(df1)-12:]
        forecast = pd.concat([test,predSArima,predProphet],axis=1)
        forecast['LSTM'] = predLSTM
        
        forecast['Ac_venda'] = test.cumsum()
        forecast['Ac_SArima'] = forecast['SARIMA'].cumsum()
        forecast['Ac_PROPHET'] = forecast['PROPHET'].cumsum()
        forecast['Ac_LSTM'] = forecast['LSTM'].cumsum()
        
        #Grava dados na tabela fato_predicao (BI)
        grava_fato_predicao(df_prod['sku'][ind],forecast)

# %%
