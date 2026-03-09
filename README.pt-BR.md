# Case Datarisk Para Cientista De Dados Junior - Extensao De Portfolio

Este repositorio amplia o case tecnico original da Datarisk e o transforma em um projeto de portfolio com app Streamlit bilingue, narrativa de decisao de negocio, carregamento de modelos via MLflow e estrategia publica de dados.

## Desafio Original

- Repositorio oficial do case: <https://github.com/datarisk-io/datarisk-case-ds-junior>
- As previsoes devem ser geradas para cada registro de `base_pagamentos_teste.csv`.
- No case, inadimplencia significa pagamento realizado com `5` dias ou mais de atraso em relacao ao vencimento.
- A submissao original foi feita primeiro. Este repositorio representa a extensao de portfolio daquela solucao.

## Politica Publica De Dados

Este repositorio nao redistribui os dados originais do case.

- Fonte oficial dos dados brutos: repositorio oficial da Datarisk.
- Modo publico do app: os CSVs brutos sao baixados em tempo de execucao a partir da fonte oficial.
- Compatibilidade local: se voce ja tiver os CSVs oficiais em `data/raw/`, o projeto usa esses arquivos primeiro.
- Cache de download: os arquivos ficam em `.cache/` para evitar downloads repetidos.

Veja `data/README.md` para a estrutura local esperada.

## Artefato Oficial Do Portfolio

O repositorio foi consolidado em torno de uma unica run final oficial:

- Modelo final: `Random Forest`
- Variante: `Sem SMOTE`
- Run oficial: `56e3d7b850c24eb78a05b2a232f30fd5`
- Metrica de selecao: `PR-AUC`
- PR-AUC: `0.7308`
- ROC-AUC: `0.9598`
- Threshold recomendado: `0.20`
- Precisao no threshold: `0.6352`
- Recall no threshold: `0.7411`
- Taxa positiva no threshold: `0.0777`

## O Que O Projeto Demonstra

- Pipeline de machine learning ponta a ponta em Python.
- Validacao temporal e comparacao de benchmark.
- Ajuste de threshold ligado a trade-offs operacionais.
- Comunicacao de negocio para recrutadores, revisores tecnicos e gestores.
- Aplicacao Streamlit preparada para demonstracao publica.

## Modo Publico Do App

O app foi estruturado para deploy publico:

- os dados brutos sao baixados do repositorio oficial da Datarisk em tempo de execucao;
- as tabelas processadas ficam em cache local depois da primeira carga;
- os artefatos analiticos ja preparados neste repositorio sao reaproveitados;
- nao ha retreinamento no ambiente de deploy;
- os binarios de modelo devem vir do MLflow para a experiencia completa de predicao.

## Como Rodar Localmente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.make_dataset
python -m src.train
python -m src.evaluate
```

Para abrir o app:

```bash
streamlit run streamlit_app.py
```

Variaveis de ambiente opcionais:

```bash
set DATARISK_DATA_SOURCE=auto
set DATARISK_DATA_REFRESH=0
set MLFLOW_TRACKING_URI=http://SEU_MLFLOW_HOST:5000
set MLFLOW_EXPERIMENT_NAME=datarisk-inadimplencia
```

## Deploy No Streamlit Community Cloud

- Arquivo principal do app: `streamlit_app.py`
- Runtime Python: `runtime.txt` -> `python-3.11`
- Dependencias: `requirements.txt`
- Comando de checagem: `python -m src.sanity_check`

Fluxo recomendado:

1. Suba este repositorio para o GitHub.
2. No Streamlit Community Cloud, crie um novo app a partir do repositorio.
3. Selecione `streamlit_app.py` como entrypoint.
4. Configure `MLFLOW_TRACKING_URI` e, se necessario, `MLFLOW_MODEL_URI` nos segredos ou variaveis do app.
5. Publique e valide as paginas depois do primeiro download dos dados.

## Link Publico Do App

Adicione o link da demonstracao aqui quando ele estiver disponivel:

- `Demo Streamlit: <adicione-o-link-aqui>`

## Checagem De Sanidade Antes Do Deploy

```bash
python -m src.sanity_check
```

O script valida:

- importacoes principais;
- disponibilidade oficial dos dados da Datarisk;
- leitura dos artefatos obrigatorios do projeto;
- compilacao basica do entrypoint do Streamlit.

## Estrutura Do Repositorio Para Deploy

```text
.
+-- app/
+-- data/
|   +-- README.md
+-- docs/
+-- models/
+-- notebooks/
+-- reports/
+-- src/
+-- tests/
+-- requirements.txt
+-- runtime.txt
+-- streamlit_app.py
```

## Riscos Antes Da Publicacao

- A pagina de predicao publica depende de conectividade com o MLflow se voce quiser o scoring completo ativo.
- A primeira carga no Streamlit Cloud sera mais lenta porque os dados brutos serao baixados e colocados em cache.
- Se o repositorio oficial da Datarisk mudar nomes ou caminhos de arquivos, o loader remoto precisara ser atualizado.

## Documentos Adicionais

- Model card em ingles: `MODEL_CARD.md`
- Model card em portugues: `MODEL_CARD.pt-BR.md`
- Resumo executivo em ingles: `docs/executive_summary.md`
- Resumo executivo em portugues: `docs/executive_summary.pt-BR.md`
