CREATE TABLE `projeto`.`fato_predicao` (
  `sk_tempo` INT NOT NULL,
  `sk_protudo` INT NOT NULL,
  `qtd_predicao` DECIMAL(18,2),
  `qtd_acumula` DECIMAL(18,2),
  PRIMARY KEY (`sk_tempo`,`sk_protudo`));
