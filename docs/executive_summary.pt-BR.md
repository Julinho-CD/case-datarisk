# Resumo Executivo

## Contexto

Este projeto amplia a submissao original do case tecnico da Datarisk para Cientista de Dados Junior e a transforma em uma aplicacao de portfolio voltada a suporte de decisao para priorizacao de risco de inadimplencia.

## Abordagem

O fluxo combina preparacao de dados, engenharia de features, validacao temporal, comparacao de benchmark entre Logistic Regression, Random Forest e LightGBM, ajuste de threshold e uma interface Streamlit bilingue para publico executivo e tecnico.
O fluxo de submissao preserva uma previsao para cada registro de `base_pagamentos_teste.csv`, e os labels de treino seguem a regra original do case: inadimplente = atraso de pagamento `>= 5` dias.

## Resultado Final

- Modelo final oficial: `Random Forest`
- Variante: `Sem SMOTE`
- Run oficial: `56e3d7b850c24eb78a05b2a232f30fd5`
- Metrica de selecao: `PR-AUC`
- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- Threshold recomendado: `0.20`

## Decisao Operacional

No threshold recomendado, o modelo sinaliza cerca de 7.8% dos casos, com precisao `0.6352` e recall `0.7411`. Na pratica, isso apoia uma fila mais focada de cobranca ou revisao de risco, priorizando os casos de maior risco sem sobrecarregar a capacidade operacional.

## Limitacoes

- Os resultados refletem o dataset do case e sua janela temporal especifica.
- Importancia de features nao implica causalidade.
- A politica de threshold deve ser recalibrada para cada contexto de negocio e capacidade do time.
- Um deploy em producao exigiria monitoramento, governanca e controles de drift.
