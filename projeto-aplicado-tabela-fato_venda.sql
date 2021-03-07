CREATE TABLE `projeto`.`fato_venda` (
  `sk_rec` INT NOT NULL,
  `sk_tempo` INT NOT NULL,
  `sk_protudo` INT NOT NULL,
  `qtd_venda` INT NOT NULL,
  `val_venda` DECIMAL(18,2) NOT NULL,
  PRIMARY KEY (`sk_rec`));
