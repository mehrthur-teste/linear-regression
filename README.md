# Linear Regression Starter

Este projeto demonstra uma implementação simples de regressão linear em Python. Ele treina um modelo, guarda-o em disco e permite fazer previsões para novos valores.

> Nota: os dados atuais são apenas de demonstração e seguem a relação artificial `y = 2x + 1`. Para usar o projeto num cenário real, substitui os dados de exemplo por dados reais e treina novamente o modelo.

## Requisitos 

- Python 3.11+
- Dependências listadas em `requirements.txt`

## Configuração

Na raiz do projeto, cria e ativa um ambiente virtual e instala as dependências:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sempre que abrires um novo terminal, ativa novamente o ambiente:

```bash
source .venv/bin/activate
```

## Treinar e guardar o modelo

Executa:

```bash
python linear_regression.py
```

O script:
- gera os dados de exemplo
- divide os dados em treino e teste
- treina um modelo `LinearRegression`
- calcula métricas como MAE, MSE e R²
- guarda o modelo em `linear_regression.joblib`

Para usar dados reais, edita a função `build_sample_data()` em `linear_regression.py` e mantém `X` e `y` com o mesmo número de linhas e a mesma estrutura esperada.

## Fazer uma previsão

Depois de treinar o modelo, executa:

```bash
python predict.py 11
```

Ou, sem argumento, o script pede o valor interativamente:

```bash
python predict.py
```

Se o ficheiro do modelo ainda não existir, executa primeiro `linear_regression.py`.

## Testes

Para validar a base do projeto, executa:

```bash
pytest
```

## Estrutura do projeto

- `linear_regression.py` — treina, avalia e guarda o modelo
- `predict.py` — carrega o modelo e produz uma previsão para um novo valor
- `requirements.txt` — dependências do projeto
- `linear_regression.joblib` — ficheiro gerado pelo treino
- `test_linear_regression.py` — testes básicos para validar o comportamento principal

## Próximos passos

- substituir os dados de exemplo por dados reais
- validar a qualidade do modelo com um conjunto apropriado
- considerar a criação de uma pipeline mais robusta para produção
