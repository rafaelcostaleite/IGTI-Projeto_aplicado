CREATE TABLE `projeto`.`dim_tempo` (
  `sk_tempo` INT NOT NULL,
  `nr_ano` INT NULL,
  `nr_mes_numero` INT NULL,
  `nr_dia_do_mes` INT NULL,
  `no_mes` VARCHAR(30) NULL,
  `nr_trimestre` INT NULL,
  `qtd_dia_mes` INT NULL,
  `ds_data_sem_hora` VARCHAR(10) NULL,
  `dt_com_hora` DATE NULL,
  `ds_mes_ano` VARCHAR(7) NULL,
  `ds_data_ymd` VARCHAR(8) NULL,
  `ds_ano_mes` VARCHAR(8) NULL,
  PRIMARY KEY (`sk_tempo`));