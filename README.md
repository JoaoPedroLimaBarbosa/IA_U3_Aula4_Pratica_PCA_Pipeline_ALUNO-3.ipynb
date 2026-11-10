# Redução de Dimensionalidade e Pipeline de Agrupamento (PCA + K-Means)

Atividade prática da disciplina de Inteligência Artificial (Unidade 3, Aula 4). O projeto aplica **PCA** e **K-Means** em indicadores socioeconômicos e de saúde de 167 países, para ajudar uma ONG fictícia (HELP International) a decidir como distribuir US$ 10 milhões em ações de combate à pobreza e apoio em desastres.

## Objetivo

Classificar os países em **três níveis de prioridade** e avaliar o efeito de cada etapa (padronização, PCA e K-Means) sobre o resultado, construindo ao final um *pipeline* que pode ser salvo e reutilizado.

## Dados

Base [*Unsupervised Learning on Country Data*](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) (Kaggle): 167 países e 9 indicadores.

| Indicador | Significado |
|---|---|
| `child_mort` | mortes de crianças menores de 5 anos a cada 1.000 nascidos vivos |
| `exports` | exportações per capita, em % do PIB per capita |
| `health` | gasto total com saúde per capita, em % do PIB per capita |
| `imports` | importações per capita, em % do PIB per capita |
| `income` | renda líquida por pessoa (US$) |
| `inflation` | taxa anual de crescimento do PIB total, em % |
| `life_expec` | expectativa de vida ao nascer, em anos |
| `total_fer` | número médio de filhos por mulher |
| `gdpp` | PIB per capita (US$) |

## Etapas do notebook

1. **Preparação e escalas:** estatísticas descritivas e verificação de valores ausentes.
2. **Efeito da padronização:** comparação do PCA com e sem `StandardScaler`.
3. **Número de componentes:** variância explicada e acumulada (80% e 90%).
4. **Interpretação dos componentes:** pesos (*loadings*) e gráfico PC1 × PC2.
5. **Pipeline de agrupamento:** `StandardScaler` → `PCA(0.9)` → `KMeans(3)`, comparação com três variações (índice de Rand ajustado) e coeficiente de silhueta para k de 2 a 6.
6. **Interpretação dos grupos:** perfil médio de cada grupo e países prioritários.
7. **Persistência:** o pipeline é salvo em `modelo_paises.pkl` e usado para classificar países novos na escala original.
8. **Complementares:** K-Means com k = 4 e detecção de países atípicos com DBSCAN.

## Principais resultados

- Sem padronização, a PC1 explica 94,8% da variância, mas só reflete `income` e `gdpp` (as maiores escalas). Com padronização, a PC1 explica 46,0% e passa a representar o **nível de desenvolvimento** do país.
- São necessários **4 componentes para 80%** e **5 para 90%** da variância. O pipeline usa 5.
- O pipeline gerou três grupos:

| Grupo | Países | Perfil |
|---|---|---|
| 0 | 36 | Alta renda e baixo risco (mortalidade infantil média de 5,0) |
| 1 | 84 | Renda média, em desenvolvimento (mortalidade de 21,9) |
| 2 | 47 | Alta vulnerabilidade (mortalidade de 93,0, renda média de US$ 3.942) |

- O PCA com 2 componentes é bom para visualização, mas altera bastante o agrupamento (ARI de 0,318), pois preserva só cerca de 63% da variância.
- A silhueta dos agrupamentos ficou em torno de 0,3 (k = 3: 0,308; k = 4: 0,329; k = 5: 0,330), o que indica grupos pouco separados. O k = 3 foi mantido por ser a necessidade da ONG.
- Países com maior mortalidade infantil no grupo prioritário: Haiti, Sierra Leone, Chad, República Centro-Africana, Mali, Nigéria, Níger, Angola, Burkina Faso e Congo (Rep. Dem.).
- O DBSCAN apontou 5 países atípicos: Luxembourg, Malta, Singapore, Qatar e Nigeria.

## Como executar

### Requisitos

- Python 3.9 ou superior
- Bibliotecas: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `joblib`, `kagglehub`

```bash
pip install numpy pandas matplotlib scikit-learn joblib kagglehub
```

### Passos

1. Clone o repositório e abra o notebook `IA_U3_Aula4_Pratica_PCA_Pipeline_ALUNO.ipynb` no VS Code, Jupyter ou Google Colab.
2. Execute as células em ordem. A base é baixada automaticamente pelo `kagglehub` (pode ser pedido login no Kaggle na primeira vez).
3. Ao final, o arquivo `modelo_paises.pkl` é gerado na pasta do notebook.

## Usando o modelo salvo

O arquivo guarda um dicionário com o pipeline, a lista ordenada de colunas e os nomes dos grupos. Como o pipeline já inclui a padronização e o PCA, os dados novos devem ser passados **na escala original**:

```python
import joblib
import pandas as pd

carregado = joblib.load("modelo_paises.pkl")

pais_novo = pd.DataFrame(
    [[60, 25, 5.0, 35, 5000, 8, 65, 3.5, 2500]],
    columns=carregado["colunas"])

grupo = carregado["modelo"].predict(pais_novo[carregado["colunas"]])[0]
print(grupo, "·", carregado["nomes"][int(grupo)])
```

> O arquivo `.pkl` depende da versão do scikit-learn usada para gerá-lo. Para evitar erros ao carregar, use a mesma versão ou gere o arquivo de novo executando o notebook.

## Limitações

- Os grupos não são bem separados (silhueta ≈ 0,3), então países na fronteira entre dois grupos podem mudar de classificação com pequenas variações nos indicadores.
- A base não informa o ano de referência dos dados nem a fonte original.
- Só 9 indicadores são considerados. Fatores como conflitos, desastres e desigualdade interna ficam de fora.
- Os resultados servem como triagem inicial e não devem ser a única base para uma decisão de distribuição de recursos.

