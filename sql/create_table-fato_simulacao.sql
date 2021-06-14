CREATE TABLE `projeto`.`fato_simulacao` (
  `sk_tempo` INT NOT NULL,
  `sk_produto` INT NOT NULL,
  `no_modelo` VARCHAR(100) NOT NULL,
  `qtd_predicao` DECIMAL(18,2),
  `qtd_acumula` DECIMAL(18,2),
  PRIMARY KEY (`sk_tempo`,`sk_protudo`,`no_modelo`));
