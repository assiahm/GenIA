import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score

# ///// Charger les données
# Charger les données
data = pd.read_json('data/reviews.jsonl', lines=True)
# Extraire les colonnes nécessaires
reviews = data[['rating', 'text']].dropna()

# Limiter les données à 200 lignes pour accélérer les tests
#reviews = reviews.head(200)
#print(reviews.head(5))

# ///// Charger le modèle et tokenizer
# Charger le modèle et le tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Détecter le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ///// Prétraitement et analyse des sentiments par lots
# Créer une classe Dataset pour gérer les données
class ReviewDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Créer le DataLoader
batch_size = 16  # Ajustez cette valeur en fonction de votre mémoire
dataset = ReviewDataset(reviews['text'].tolist())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Analyser les sentiments par lots
all_predictions = []
for batch in dataloader:
    # Tokenisation par lots
    tokens = tokenizer(batch, return_tensors='pt', truncation=True, padding=True, max_length=512)
    tokens = {key: val.to(device) for key, val in tokens.items()}
    
    # Passage dans le modèle
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        predictions = probabilities.argmax(axis=-1) + 1
        all_predictions.extend(predictions)

# Ajouter les prédictions aux données
reviews['predicted_rating'] = all_predictions

# ///// Évaluation des performances
# Calculer la corrélation de Pearson
actual_ratings = reviews['rating']
predicted_ratings = reviews['predicted_rating']

correlation, _ = pearsonr(actual_ratings, predicted_ratings)
print(f"Corrélation de Pearson : {correlation}")

# Afficher les colonnes rating, predicted_rating, et text
print(reviews[['rating', 'predicted_rating', 'text']].head(10))

# ///// Visualisation des résultats
# Créer une matrice de corrélation
# Calculer la matrice de corrélation
correlation_matrix = reviews[['rating', 'predicted_rating']].corr()

# Ajouter des informations dans la matrice de corrélation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title(f"Matrice de Corrélation\nNombre d'avis : {len(reviews)}\nCorrélation : {correlation:.2f}")
plt.xlabel("Colonnes")
plt.ylabel("Colonnes")
plt.show()

# ///// Matrice de confusion
# Calculer la matrice de confusion
confusion_matrix = pd.crosstab(reviews['rating'], reviews['predicted_rating'], rownames=['Valeurs réelles'], colnames=['Valeurs prédites'], normalize=False)

# Visualisation de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title("Matrice de Confusion entre Notes Réelles et Prédites")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Valeurs Réelles")
plt.show()


# Graphique comparatif des valeurs réelles et prédites
plt.figure(figsize=(10, 6))
bar_width = 0.4  # Largeur des barres

# Calcul des fréquences pour chaque note
ratings_counts = reviews['rating'].value_counts().sort_index()
predicted_counts = reviews['predicted_rating'].value_counts().sort_index()

# Position des barres sur l'axe des x
x_real = [x - bar_width / 2 for x in ratings_counts.index]
x_predicted = [x + bar_width / 2 for x in predicted_counts.index]

# Tracer les barres pour les valeurs réelles et prédites
plt.bar(x_real, ratings_counts, width=bar_width, label='Valeurs Réelles', alpha=0.7, color='blue')
plt.bar(x_predicted, predicted_counts, width=bar_width, label='Valeurs Prédites', alpha=0.7, color='orange')

# Ajouter des étiquettes, une légende et un titre
plt.xlabel('Notes (1 à 5)', fontsize=12)
plt.ylabel('Nombre d\'Avis', fontsize=12)
plt.title('Comparaison des Notes Réelles et Prédites', fontsize=14)
plt.xticks([1, 2, 3, 4, 5])  # Ajuster les ticks de l'axe x
plt.legend()
plt.tight_layout()

# Afficher le graphique
plt.show()

rmse = mean_squared_error(actual_ratings, predicted_ratings, squared=False)
accuracy = accuracy_score(actual_ratings, predicted_ratings)
print(f"RMSE : {rmse}, Accuracy : {accuracy}")