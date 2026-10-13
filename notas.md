# Notas de interpretabilidade em ML: previsão de inadimplência

Dataset: Loan Default (Kaggle), 255.347 linhas, 16 variáveis de entrada e o alvo `Default`. Modelos: KNN, Naïve Bayes (Gaussiano) e Árvore de Decisão.

## 1. Dados e pré-processamento

- Divisão 80/20 estratificada (`random_state=42`): 204.277 linhas de treino e 51.070 de teste.
- Numéricas (9) padronizadas com `StandardScaler`, necessário pro KNN. Categóricas (7) viraram colunas 0/1 com `OneHotEncoder`, o que dá 31 colunas no total.
- O dataset é bem desbalanceado: **88,4%** não deram default e **11,6%** deram. A proporção se manteve igual no treino e no teste.

Por isso olhamos principalmente recall, F1 da classe 1 e AUC. A acurácia (~88%) engana: um modelo que sempre diz "não vai dar default" já acerta isso.

## 2. Resultados dos modelos (limiar padrão)

| Modelo | Recall (default) | Precisão (default) | F1 (default) | AUC |
|---|---|---|---|---|
| KNN (k=15, `distance`) | 0,02 | 0,50 | 0,05 | 0,677 |
| Naïve Bayes (prior padrão) | 0,05 | 0,54 | 0,09 | 0,746 |
| Naïve Bayes (prior 50/50) | 0,70 | 0,21 | 0,33 | 0,746 |
| Árvore (profundidade 5) | 0,02 | 0,63 | 0,04 | 0,716 |
| Árvore (profundidade 5, `balanced`) | 0,66 | 0,21 | 0,31 | 0,719 |

Dois pontos pra não esquecer:

- A AUC do NB é a mesma nas duas versões porque mudar o prior só desloca o limiar de decisão, não muda a ordenação dos clientes por risco. O mesmo vale pra comparar os dois NB.
- Balancear (prior no NB, `class_weight` na árvore) sobe o recall de ~0,02 pra ~0,66-0,70, mas a precisão cai pra ~0,21. Ou seja, o modelo passa a achar quase 7 em cada 10 inadimplentes, ao custo de marcar muita gente que não daria default. Qual lado é melhor depende do custo de cada erro, o que é uma decisão de negócio.

## 3. KNN e desbalanceamento

O KNN decide pela maioria dos vizinhos mais próximos. Como ~88% dos registros são classe 0, é de se esperar que a vizinhança da maioria dos pontos seja dominada por não-defaults.

Testamos duas formas de ponderar os vizinhos:

| Configuração | Recall (classe 1) | F1 (classe 1) |
|---|---|---|
| `weights='uniform'` | 0,02 | 0,04 |
| `weights='distance'` | 0,02 | 0,05 |

Matriz de confusão com `uniform`: [[45002, 137], [5792, 139]]

Matriz de confusão com `distance`: [[44999, 140], [5790, 141]]

Leitura: linhas = classe real (0 = não-default, 1 = default), colunas = classe prevista. Dos 5.931 defaults reais, o modelo só acertou 141.

Trocar `weights` quase não mudou nada. O que isso indica é que o problema provavelmente não está em como os votos são ponderados, mas em quem aparece entre os vizinhos. **É uma hipótese**: não olhamos diretamente a composição dos vizinhos.

**O que não foi testado** (e portanto não dá pra afirmar que o KNN "não consegue"): ajustar o limiar de decisão usando `predict_proba`, undersampling/oversampling, mudar o k (usamos só k=15, sem tuning) e remover colunas redundantes. Ao contrário do NB (`priors`) e da árvore (`class_weight`), o KNN não tem um parâmetro nativo de balanceamento.

Observação: o `GaussianNB` também **não** tem `class_weight`. O que ele permite é definir manualmente as probabilidades a priori (`priors`), que foi o que fizemos.

## 4. Interpretabilidade

### 4.1 Árvore de Decisão

Importância das variáveis (árvore balanceada): Age 0,43, InterestRate 0,20, Income 0,20, MonthsEmployed 0,10, LoanAmount 0,07. **Todas as outras ficaram com importância 0**, inclusive CreditScore.

Pelo desenho da árvore (`arvore.png`, 3 níveis), a primeira divisão é por idade: quem tem **≤ ~42,5 anos** cai majoritariamente em "default". Depois a árvore refina por juros (≤ ~12,7%), renda (≤ ~37,6 mil), valor do empréstimo (≤ ~90 mil) e tempo de emprego (≤ ~38,5 meses).

Cuidados ao ler a figura:
- Os limiares estão em z-score na figura. Convertemos pra unidade real com média e desvio do treino (valor real = z × desvio + média).
- O campo `value` está **ponderado** (por causa do `class_weight`), então não são contagens reais. Só `samples` é real.
- A árvore é fraca: os nós da base têm gini entre 0,34 e 0,50, e o melhor ainda tem ~25% de default. Isso combina com a AUC de ~0,72.

### 4.2 Naïve Bayes (leitura direta dos parâmetros)

Comparamos a diferença das médias entre as classes (padronizadas) dividida pelo desvio:

| Variável | Separação | Em quem deu default... |
|---|---|---|
| Age | 0,54 | é menor |
| InterestRate | 0,42 | é maior |
| MonthsEmployed | 0,30 | é menor |
| Income | 0,30 | é menor |
| LoanAmount | 0,27 | é maior |

Perfil: inadimplentes são mais jovens, com menos renda, menos tempo de emprego, empréstimos maiores e juros mais altos.

### 4.3 Permutation importance (queda na AUC ao embaralhar cada coluna)

| Variável | KNN | NB | Árvore |
|---|---|---|---|
| Age | 0,060 | 0,081 | 0,095 |
| InterestRate | 0,033 | 0,046 | 0,046 |
| Income | 0,039 | 0,023 | 0,040 |
| MonthsEmployed | 0,017 | 0,023 | 0,020 |
| LoanAmount | 0,013 | 0,018 | 0,018 |
| CreditScore | 0,020 | 0,003 | 0,000 |

Notas: o KNN usou só 5.000 linhas do teste e 3 repetições (ele é lento). NB e árvore usaram o teste inteiro e 5 repetições. As 5 variáveis do topo são as mesmas nos três modelos. A única divergência real é o **CreditScore**: o KNN usa, a árvore ignora e o NB quase ignora.

### 4.4 SHAP no KNN (`shap_knn.png`)

- KernelExplainer, **100 linhas** do teste, fundo de 10 pontos (k-means) e `nsamples=300`.
- Top 5 (média do |SHAP|): Age 0,032, InterestRate 0,027, MonthsEmployed 0,022, Income 0,016, LoanAmount 0,016. O top 5 foi idêntico em duas execuções; da 6ª posição pra baixo as ordens trocaram, então **só o topo é confiável**.
- Direção (gráfico): idade baixa, juros altos, pouco tempo de emprego, renda baixa, empréstimo alto e score baixo empurram a previsão pra default. Bate com o NB.
- Os valores são pequenos (±0,1 em probabilidade) porque o KNN raramente dá probabilidades altas. Vale comparar ranking e direção, não magnitude.

### 4.5 LIME no KNN

Explicamos dois casos: o primeiro falso positivo e o primeiro falso negativo (escolhidos sem critério, são exemplos, não amostra).

- **Falso positivo** (prob. de default segundo o KNN = 0,53, quase no limiar): juros altos, renda baixa, empréstimo alto e pouco tempo de emprego.
- **Falso negativo** (prob. = 0,06, mas o cliente deu default): idade ≤ ~31 anos, empréstimo > ~189 mil e < ~30 meses de emprego, todos sinais de risco nos outros métodos, e mesmo assim o KNN previu baixo risco.
- O R² do modelo local do LIME foi **0,20 e 0,17**, ou seja, as explicações são aproximações de baixa fidelidade. Servem como indício coerente com o resultado global, não como prova do que o KNN fez.

## 5. Limitações

- Colunas binárias (Yes/No) viram duas colunas espelhadas. Isso conta o mesmo efeito duas vezes no NB e no KNN e aparece duplicado no SHAP e no ranking do NB (`HasCoSigner_Yes` e `HasCoSigner_No`).
- O `GaussianNB` assume distribuição normal, o que não vale pras colunas 0/1.
- Árvore com profundidade fixa em 5 (não otimizamos), KNN com k=15 (sem tuning).
- SHAP com poucas amostras e `nsamples=300`; LIME com R² baixo e só 2 casos.
- Variáveis correlacionadas dividem a importância entre si, então rankings individuais devem ser lidos com cuidado.
- Os métodos concordam entre si, mas isso mostra robustez do sinal, não que o modelo esteja certo.

## 6. Pontos para a Etapa 4

- Todos os modelos, no limiar padrão, quase não detectam defaults. Balancear resolve o recall, não a precisão.
- Em AUC: NB (0,746) > Árvore (0,716-0,719) > KNN (0,677).
- As mesmas 5 variáveis (Age, InterestRate, Income, MonthsEmployed, LoanAmount) lideram em todos os métodos de interpretação, com a mesma direção.
- CreditScore é a divergência: só o KNN usa. Hipóteses a testar: a árvore rasa não o escolhe, ou o efeito é fraco e só a distância do KNN o capta.
- Interpretabilidade: árvore (regras legíveis) e NB (parâmetros diretos) são interpretáveis por construção. KNN precisa de SHAP/LIME, que são aproximações caras e ruidosas.