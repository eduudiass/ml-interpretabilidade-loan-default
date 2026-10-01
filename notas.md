# Notas de interpretabilidade em ML: previsão de inadimplência

## KNN e desbalanceamento das classes

O dataset utilizado apresenta um forte desbalanceamento entre as classes:

- **88,4%** dos registros pertencem à classe 0, que representa clientes que não entraram em default.
- **11,6%** pertencem à classe 1, que representa clientes que entraram em default.

Esse desequilíbrio é importante porque o KNN classifica cada novo caso com base na maioria dos seus vizinhos mais próximos. Como a classe 0 é muito mais frequente, existe uma tendência de que a maioria dos vizinhos pertença a essa classe, mesmo quando o cliente deveria ser classificado como inadimplente.

## Testes realizados
Foram testadas duas formas de calcular a influência dos vizinhos:

| Configuração                   | Recall (classe 1) | F1-score (classe 1) |
|---------------------------------|--------------------|-----------------------|
| `weights='uniform'` (padrão)   | 0,02               | 0,04                  |
| `weights='distance'`           | 0,02               | 0,05                  |
Na configuração `uniform`, todos os vizinhos têm o mesmo peso na decisão. Já em `distance`, os vizinhos mais próximos recebem maior influência.

### Matriz de confusão — `weights='uniform'`
[[45002 137]
[ 5792 139]]

### Matriz de confusão — `weights='distance'`
[[44999 140]
[ 5790 141]]
Leitura da matriz: linha 0 = casos reais de "não-default", linha 1 = casos reais de "default". Coluna 0 = previsto "não-default", coluna 1 = previsto "default". Ou seja, na segunda configuração, de 5.931 casos reais de default (5790+141), o modelo só identificou 141 corretamente.

## Interpretação dos resultados
A utilização de `weights='distance'` praticamente não alterou o desempenho do modelo na identificação da classe minoritária. O recall permaneceu em 0,02, enquanto o F1-score passou apenas de 0,04 para 0,05.
Isso sugere que o principal problema não está na forma como os votos dos vizinhos são ponderados, mas na própria composição do conjunto de treinamento. Como aproximadamente 88% dos registros pertencem à classe 0, a vizinhança da maioria dos pontos de teste tende a ser dominada por clientes que não entraram em default.
Consequentemente, mesmo dando mais peso aos vizinhos próximos, o modelo continua encontrando predominantemente exemplos da classe 0. Por isso, a alteração em `weights` não foi suficiente para melhorar a identificação dos casos de default.

## Comparação com outros modelos
Esse comportamento é diferente do observado em modelos como Naïve Bayes e Árvore de Decisão. No scikit-learn, esses algoritmos permitem utilizar o parâmetro `class_weight="balanced"`, que ajusta o peso das classes durante o treinamento.
O KNN não possui um mecanismo nativo equivalente. Uma alternativa seria balancear os dados antes do treinamento, utilizando técnicas como undersampling ou oversampling. No entanto, essas abordagens não foram testadas neste trabalho por causa da limitação de tempo.

## Conclusão para a Etapa 4
Os resultados indicam que o KNN apresenta uma limitação importante quando aplicado a datasets com classes muito desbalanceadas. Como sua decisão depende da maioria local entre os vizinhos, a predominância da classe 0 dificulta a identificação dos casos de default.
Além disso, os ajustes testados, como `n_neighbors` e `weights`, não foram suficientes para resolver esse problema. Portanto, o baixo desempenho do KNN nesse cenário parece estar relacionado a uma limitação estrutural do algoritmo diante do desbalanceamento, e não a um erro de configuração do modelo.