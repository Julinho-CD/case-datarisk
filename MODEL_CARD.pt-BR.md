# Model Card - Score de Risco de Inadimplencia

## Objetivo do Modelo

Prever a probabilidade de inadimplencia para apoiar a priorizacao operacional em cobranca e analise de risco.

Este modelo e ferramenta de suporte a decisao, nao mecanismo autonomo de aprovacao/rejeicao.

## Contexto do Desafio

Construido a partir do case tecnico de Cientista de Dados Junior da Datarisk e evoluido como projeto de portfolio.

## Definicao do Target

- Coluna alvo: `PROBABILIDADE_INADIMPLENCIA`
- Regra binaria (treino): inadimplente = `1` se atraso no pagamento for `>= 5` dias, senao `0`

## Dados Utilizados

- Base cadastral (`base_cadastral.csv`)
- Base mensal de contexto financeiro (`base_info.csv`)
- Historico de pagamentos de desenvolvimento (`base_pagamentos_desenvolvimento.csv`)
- Base de scoring (`base_pagamentos_teste.csv`)

Os datasets nao sao redistribuidos neste repositorio.

## Principais Grupos de Features

- Atributos de fatura/pagamento (`VALOR_A_PAGAR`, `TAXA`)
- Perfil de cliente (`PORTE`, `SEGMENTO_INDUSTRIAL`, `DOMINIO_EMAIL`, `CEP_2_DIG`)
- Features temporais derivadas (`TEMPO_CADASTRO`, `PRAZO_EMISSAO_VENCIMENTO`, `MES`)
- Features historicas (`QTDE_ATRASOS_ANT`, `TICKET_MEDIO_ANT`)
- Proxy de capacidade de pagamento (`VALOR_RELATIVO_RENDA`)

## Selecao do Modelo

O benchmark avaliou variantes de Regressao Logistica, Random Forest e LightGBM.

Melhor run (criterio PR-AUC):

- Modelo: Random Forest (sem SMOTE)
- Run ID: `56e3d7b850c24eb78a05b2a232f30fd5`

## Metricas-Chave (Validacao)

- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- F1 @ 0.50: `0.5629`
- Melhor threshold: `0.20`
- F1 @ melhor threshold: `0.6841`
- Precisao @ melhor threshold: `0.6352`
- Recall @ melhor threshold: `0.7411`
- Taxa positiva @ melhor threshold: `0.0777`

## Uso Operacional Recomendado

1. Pontuar todas as faturas/clientes elegiveis.
2. Aplicar politica de threshold (`~0.20` como referencia inicial).
3. Priorizar fila de maior risco para acao manual/assistida.
4. Recalibrar threshold conforme capacidade da equipe e custo de falso negativo.

## Trade-offs de Erro

- Falso Negativo (alto risco nao sinalizado): atraso de cobranca e perda potencial.
- Falso Positivo (baixo risco sinalizado): esforco operacional desnecessario.

A escolha de threshold deve equilibrar explicitamente esses custos.

## Limitacoes

- Treinado em dados do desafio com janelas temporais especificas.
- A performance pode degradar sob mudanca de distribuicao.
- Importancia de features indica associacao, nao causalidade.
- Binarios locais pesados nao ficam neste repositorio por estrategia MLflow-first.

## Riscos e Cuidados

- Nao usar score como criterio unico para decisoes irreversiveis.
- Monitorar fairness e performance por segmento antes de producao.
- Acompanhar drift de calibracao e retreinar periodicamente.

## Quando Usar

- Priorizacao de filas de cobranca/risco.
- Planejamento de capacidade operacional com recursos limitados.

## Quando Nao Usar

- Acoes punitivas totalmente automaticas sem revisao humana.
- Contextos com exigencias legais/compliance sem controles adicionais de explicabilidade.
