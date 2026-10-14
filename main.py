import pandas as pd

# carrega o CSV e transforma em DataFrame
df = pd.read_csv("Loan_default.csv")

# tamanho do dataset
print(df.shape)

# nomes das colunas
print(df.columns.tolist())

# valores nulos por coluna
print(df.isnull().sum())

# proporção de cada classe do target
print(df["Default"].value_counts(normalize=True))

#Bilbiotecas para a realização dos gráficos
import matplotlib.pyplot as plt
import seaborn as sns

# Salva os gráficos de análise exploratória

# 1. Gráfico de contagem da variável alvo (evidencia a proporção de 88% / 12%)
cores_default = {0: 'blue', 1: 'red', '0': 'blue', '1': 'red'}

# 1. Gráfico de contagem da variável alvo
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Default', hue='Default', palette=cores_default, legend=False)
plt.title('Distribuição da Variável Alvo (Default)')
plt.savefig('distribuicao_classes.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Histograma comparando a distribuição de Idade
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Age', hue='Default', bins=30, kde=True, palette=cores_default)
plt.title('Distribuição de Idade por Default')
plt.savefig('histograma_idade.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Histograma comparando a distribuição de Renda (Income)
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x='Income', hue='Default', bins=30, kde=True, palette=cores_default)
plt.title('Distribuição de Renda por Default')
plt.savefig('histograma_rendimento.png', dpi=150, bbox_inches='tight')
plt.close()

from sklearn.model_selection import train_test_split

# colunas de entrada. LoanID fica de fora,
# Default fica de fora 
feature_cols = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
                 'Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                 'HasDependents', 'LoanPurpose', 'HasCoSigner']

X = df[feature_cols]   # features
y = df['Default']      # target

# divide em treino (80%) e teste (20%)
# stratify mantém a proporção de default nos dois conjuntos
# random_state fixa a aleatoriedade, pra poder reproduzir depois
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# confere o tamanho de cada conjunto
print(X_train.shape, X_test.shape)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# numéricas: padroniza a escala (necessário pro KNN)
colunas_numericas = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
                      'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio']

# categóricas: transforma texto em colunas binárias (0/1)
colunas_categoricas = ['Education', 'EmploymentType', 'MaritalStatus', 'HasMortgage',
                        'HasDependents', 'LoanPurpose', 'HasCoSigner']

# aplica cada transformação no grupo de colunas certo
preprocessador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OneHotEncoder(), colunas_categoricas)
    ]
)

# aprende e transforma no treino
X_train_transformado = preprocessador.fit_transform(X_train)

# só transforma no teste (sem aprender de novo)
X_test_transformado = preprocessador.transform(X_test)

# confere o novo formato (mais colunas por causa do OneHotEncoder)
print(X_train_transformado.shape)

# confere se a proporção de default se manteve em treino e teste
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# cria o modelo KNN com k=15 vizinhos
modelo_knn = KNeighborsClassifier(n_neighbors=15, weights='distance')

# treina (KNN só guarda os dados, não calcula nada ainda)
modelo_knn.fit(X_train_transformado, y_train)

# gera as previsões pro conjunto de teste
y_pred_knn = modelo_knn.predict(X_test_transformado)

# compara previsão com realidade
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

from sklearn.naive_bayes import GaussianNB

# NB padrão: usa a proporção do treino (88%/12%) como prior
modelo_nb = GaussianNB()
modelo_nb.fit(X_train_transformado, y_train)
y_pred_nb = modelo_nb.predict(X_test_transformado)

print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# NB com prior igual pras duas classes (50%/50%)
modelo_nb2 = GaussianNB(priors=[0.5, 0.5])
modelo_nb2.fit(X_train_transformado, y_train)
y_pred_nb2 = modelo_nb2.predict(X_test_transformado)

print(confusion_matrix(y_test, y_pred_nb2))
print(classification_report(y_test, y_pred_nb2))

from sklearn.tree import DecisionTreeClassifier

# árvore sem balanceamento
modelo_arvore = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arvore.fit(X_train_transformado, y_train)
y_pred_arvore = modelo_arvore.predict(X_test_transformado)

print(confusion_matrix(y_test, y_pred_arvore))
print(classification_report(y_test, y_pred_arvore))

# árvore com pesos balanceados (compensa o desbalanceamento)
modelo_arvore2 = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
modelo_arvore2.fit(X_train_transformado, y_train)
y_pred_arvore2 = modelo_arvore2.predict(X_test_transformado)

print(confusion_matrix(y_test, y_pred_arvore2))
print(classification_report(y_test, y_pred_arvore2))

from sklearn.metrics import roc_auc_score

# AUC: compara a qualidade do ranking de risco, sem depender de limiar ou prior
print("AUC KNN:", roc_auc_score(y_test, modelo_knn.predict_proba(X_test_transformado)[:, 1]))
print("AUC NB:", roc_auc_score(y_test, modelo_nb.predict_proba(X_test_transformado)[:, 1]))
print("AUC Árvore:", roc_auc_score(y_test, modelo_arvore.predict_proba(X_test_transformado)[:, 1]))
print("AUC Árvore balanceada:", roc_auc_score(y_test, modelo_arvore2.predict_proba(X_test_transformado)[:, 1]))

from sklearn.tree import plot_tree

# nomes das colunas depois do pré-processamento (31 colunas)
nomes = preprocessador.get_feature_names_out()

# importância de cada feature na árvore balanceada
importancias = pd.Series(modelo_arvore2.feature_importances_, index=nomes).sort_values(ascending=False)
print(importancias.head(10))

# salva o desenho da árvore (3 níveis, pra ficar legível)
plt.figure(figsize=(24, 10))
plot_tree(modelo_arvore2, feature_names=nomes, class_names=['Não default', 'Default'],
          filled=True, max_depth=3, fontsize=8)
plt.savefig("arvore.png", dpi=150, bbox_inches='tight')
plt.close()  # fecha a figura da árvore, senão o SHAP desenha por cima dela

# valores reais dos limiares da árvore (a árvore usa z-score)
# valor real = z * desvio + média
escala = preprocessador.named_transformers_['num']
print("Médias:", dict(zip(colunas_numericas, escala.mean_.round(2))))
print("Desvios:", dict(zip(colunas_numericas, escala.scale_.round(2))))

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# junta pré-processamento + modelo, pra embaralhar as 16 colunas originais (e não as 31 do OneHot)
pipe_knn = Pipeline([('prep', preprocessador), ('modelo', modelo_knn)])
pipe_nb = Pipeline([('prep', preprocessador), ('modelo', modelo_nb)])
pipe_arvore = Pipeline([('prep', preprocessador), ('modelo', modelo_arvore2)])

# embaralha cada coluna e mede quanto o AUC cai (quanto mais cai, mais importante a coluna)
def importancia(pipe, X_aval, y_aval, repeticoes):
    r = permutation_importance(pipe, X_aval, y_aval, scoring='roc_auc',
                               n_repeats=repeticoes, random_state=42)
    return pd.Series(r.importances_mean, index=X_aval.columns).sort_values(ascending=False)

# KNN é lento, então usa só 5000 linhas do teste (precisa citar isso no relatório)
X_teste_pequeno = X_test.sample(5000, random_state=42)
y_teste_pequeno = y_test.loc[X_teste_pequeno.index]

print("Permutation KNN:")
print(importancia(pipe_knn, X_teste_pequeno, y_teste_pequeno, 3))

print("Permutation NB:")
print(importancia(pipe_nb, X_test, y_test, 5))

print("Permutation Árvore:")
print(importancia(pipe_arvore, X_test, y_test, 5))

import numpy as np

# médias e variâncias de cada coluna (já padronizada) por classe, segundo o NB
medias = pd.DataFrame(modelo_nb.theta_, columns=nomes, index=['Não default', 'Default'])
variancias = pd.DataFrame(modelo_nb.var_, columns=nomes, index=['Não default', 'Default'])

# diferença entre as médias das classes, dividida pelo desvio padrão
# (quanto maior, mais a coluna separa quem dá default de quem não dá)
diferenca = medias.loc['Default'] - medias.loc['Não default']
separacao = (diferenca.abs() / np.sqrt((variancias.loc['Default'] + variancias.loc['Não default']) / 2)).sort_values(ascending=False)
print(separacao.head(10))

# sentido da diferença nas 5 primeiras (positivo = valor maior em quem deu default)
print(diferenca.loc[separacao.head(5).index])

import shap
from lime.lime_tabular import LimeTabularExplainer

# sorteia 100 linhas do teste mantendo a proporção de default (SHAP é lento no KNN)
indices, _ = train_test_split(np.arange(len(y_test)), train_size=100, stratify=y_test, random_state=42)
amostra = X_test_transformado[indices]

# resume o treino em 10 pontos médios (fundo de comparação do SHAP)
fundo = shap.kmeans(X_train_transformado, 10)

# SHAP: quanto cada coluna empurra a probabilidade de default pra cima ou pra baixo
explicador_shap = shap.KernelExplainer(lambda x: modelo_knn.predict_proba(x)[:, 1], fundo)
valores_shap = explicador_shap.shap_values(amostra, nsamples=300)

# salva os valores pra não precisar recalcular se o gráfico der problema
np.save("valores_shap.npy", valores_shap)

# importância média (valor absoluto) de cada coluna
importancia_shap = pd.Series(np.abs(valores_shap).mean(axis=0), index=nomes).sort_values(ascending=False)
print(importancia_shap.head(10))

# salva o gráfico resumo
shap.summary_plot(valores_shap, amostra, feature_names=list(nomes), show=False)
plt.savefig("shap_knn.png", dpi=150, bbox_inches='tight')
plt.close()

# LIME: explica casos individuais. Pega um falso positivo e um falso negativo do KNN
real = y_test.to_numpy()
falso_positivo = np.where((real == 0) & (y_pred_knn == 1))[0][0]
falso_negativo = np.where((real == 1) & (y_pred_knn == 0))[0][0]

# as 22 últimas colunas (OneHot) são categóricas
explicador_lime = LimeTabularExplainer(X_train_transformado, feature_names=list(nomes),
                                       class_names=['Não default', 'Default'], mode='classification',
                                       categorical_features=list(range(9, 31)), random_state=42)

for nome, i in [('Falso positivo', falso_positivo), ('Falso negativo', falso_negativo)]:
    exp = explicador_lime.explain_instance(X_test_transformado[i], modelo_knn.predict_proba, num_features=8)
    print(nome, "- probabilidade de default segundo o KNN:", exp.predict_proba[1])
    print(exp.as_list())
    # R² do modelo local do LIME (perto de 0 = explicação pouco confiável)
    print("R² local do LIME:", exp.score)

# diagnóstico: o que o KNN enxergou na vizinhança desses dois casos?
# pega os 15 vizinhos mais próximos no treino e olha o rótulo real de cada um (1 = default)
y_treino = y_train.to_numpy()
for nome, i in [('Falso positivo', falso_positivo), ('Falso negativo', falso_negativo)]:
    distancias, vizinhos = modelo_knn.kneighbors(X_test_transformado[i].reshape(1, -1))
    rotulos = y_treino[vizinhos[0]]
    print(nome, "- rótulos dos 15 vizinhos:", rotulos, "| defaults:", rotulos.sum(), "de 15")