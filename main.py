import pandas as pd

# carrega o CSV e transforma em DataFrame
df = pd.read_csv("raw/Loan_default.csv")

# tamanho do dataset
print(df.shape)

# nomes das colunas
print(df.columns.tolist())

# valores nulos por coluna
print(df.isnull().sum())

# proporção de cada classe do target
print(df["Default"].value_counts(normalize=True))

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