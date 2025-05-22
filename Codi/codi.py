import pandas as pd
df = pd.read_csv("reddit_trending_2025-04-08.csv", sep=';', lineterminator='\n')
# Nuls
df.isnull().sum()

# Buits (cadenes buides)
(df == "").sum()

# Eliminar files amb valors nuls o buits en columnes crítiques
df_clean = df.dropna(subset=["Títol", "Subreddit", "Autor", "Enllaç\r"])
df_clean = df_clean[(df_clean["Títol"] != "") & (df_clean["Subreddit"] != "")]

# Opcional: revisar si cal eliminar vots iguals a 0
df_clean = df_clean[df_clean["Vots"] > 0]
df_clean["Posició"] = pd.to_numeric(df_clean["Posició"], errors="coerce")
df_clean["Vots"] = pd.to_numeric(df_clean["Vots"], errors="coerce")
df_clean["Subreddit"] = df_clean["Subreddit"].astype("category")


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df_clean["Vots"])
plt.title("Boxplot de vots")
plt.show()

# Eliminar duplicats
df_clean = df_clean.drop_duplicates()

# Comprovar que les URL de la variable Enllaç comencen per https://www.reddit.com/.
df_clean = df_clean[df_clean["Enllaç\r"].str.startswith("https://www.reddit.com/")]
df_clean.to_csv('reddit_trending_clean.csv', index=False, sep=';')

# Model supervisat:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Dataset carregat prèviament
df['Vots'] = pd.to_numeric(df['Vots'], errors='coerce')
df = df.dropna(subset=['Vots'])

# Crear una variable binària (per sobre de la mitjana)
threshold = df['Vots'].mean()
df['AltaPopularitat'] = (df['Vots'] > threshold).astype(int)

# Codificació de variables categòriques
df_encoded = pd.get_dummies(df[['Subreddit', 'Autor']], drop_first=True)

# Dades d'entrada i sortida
X = df_encoded
y = df['AltaPopularitat']

# Separar entrenament i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Resultats
y_pred = model.predict(X_test)
print("Informe de classificació:\n", classification_report(y_test, y_pred))

# Model no supervisat:
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Agrupació basada en vots i subreddit codificat
df_cluster = df[['Vots', 'Subreddit']]
df_cluster = df_cluster.dropna()

# Codificació
df_cluster_encoded = pd.get_dummies(df_cluster, columns=['Subreddit'], drop_first=True)

# Escalat
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cluster_encoded)

# Model KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualització bàsica (si només es volen veure agrupaments)
plt.scatter(df['Vots'], df['Cluster'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Vots")
plt.ylabel("Grup (clúster)")
plt.title("Agrupació de publicacions segons vots i subreddit")
plt.show()

# Trobar el subreddit més popular:
subreddit_popular = df['Subreddit'].value_counts().idxmax()
df['EsPopular'] = (df['Subreddit'] == subreddit_popular).astype(int)

# Proves prèvies:
from scipy.stats import shapiro, levene

# Separació de grups
vots_popular = df[df['EsPopular'] == 1]['Vots']
vots_resta = df[df['EsPopular'] == 0]['Vots']

# Normalitat
print("Shapiro subreddit popular:", shapiro(vots_popular))
print("Shapiro resta:", shapiro(vots_resta))

# Homogeneïtat de variàncies
print("Levene test:", levene(vots_popular, vots_resta))

# Mann-Whitney U test:
from scipy.stats import mannwhitneyu

# vots_popular = vots del subreddit que vols comparar
# vots_resta = vots dels altres subreddits

stat, p = mannwhitneyu(vots_popular, vots_resta, alternative='two-sided')
print("Mann-Whitney U Test:")
print("Estadístic:", stat)
print("p-valor:", p)
