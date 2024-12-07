# GenIA

Topic Modeling et Analyse des Sentiments des Avis Clients
Description du Projet
Ce projet vise à analyser les avis clients provenant du dataset Amazon Review Dataset 2023 pour :

Identifier les thèmes principaux à l'aide de techniques de clustering non supervisé.
Analyser les sentiments des avis clients grâce à un modèle pré-entraîné.
Le projet est structuré en trois étapes principales :

Prétraitement des Données
Clustering Non Supervisé pour Identifier des Topics
Analyse des Sentiments
Structure du Répertoire
bash
Copy code
.
├── data/
│   ├── reviews.jsonl                 # Fichier brut contenant les avis
│   ├── preprocessed_reviews.json     # Fichier JSON des avis nettoyés
│
├── src/
│   ├── preprocessing.py              # Script pour le prétraitement des données
│   ├── clustering.py                 # Script pour le clustering des avis
│   ├── sentiment_analysis.py         # Script pour l'analyse des sentiments
│
├── reports/
│   ├── Rapport_Projet.pdf            # Rapport détaillé du projet
│
├── env/                              # Environnement virtuel (optionnel)
│
├── requirements.txt                  # Liste des dépendances Python
├── README.md                         # Description du projet
Prérequis
Python : Version >= 3.8
GPU (optionnel) : Pour accélérer l'exécution des modèles pré-entraînés.
Installation
Clonez le dépôt :

bash
Copy code
git clone https://github.com/votre-repo/topic-modeling-sentiment-analysis.git
cd topic-modeling-sentiment-analysis
Installez l'environnement virtuel (optionnel) :

bash
Copy code
python -m venv env
source env/bin/activate  # Sur Mac/Linux
env\Scripts\activate     # Sur Windows
Installez les dépendances :

bash
Copy code
pip install -r requirements.txt
Téléchargez les modèles nécessaires via SpaCy et Hugging Face.

Instructions d'Exécution
1. Prétraitement des Données
Lancez le script de prétraitement pour nettoyer et tokeniser les avis :

bash
Copy code
python src/preprocessing.py
Sortie :

Un fichier JSON (preprocessed_reviews.json) contenant les tokens nettoyés pour chaque avis.
2. Clustering Non Supervisé
Appliquez un clustering KMeans pour identifier les thèmes principaux :

bash
Copy code
python src/clustering.py
Sorties :

Mots-clés par cluster.
Méthode Elbow et Silhouette Score pour choisir le nombre de clusters.
Rapport visuel dans les graphiques générés.
3. Analyse des Sentiments
Analysez les sentiments des avis en utilisant un modèle BERT pré-entraîné :

bash
Copy code
python src/sentiment_analysis.py
Sorties :

Corrélation entre les notes réelles et prédites.
Matrices de confusion et graphiques comparant les distributions.
Dépendances
Transformers : Pour les modèles pré-entraînés (Hugging Face).
SpaCy : Pour le prétraitement des textes.
Matplotlib/Seaborn : Pour les visualisations.
Scikit-learn : Pour le clustering et l'évaluation.
Pandas : Pour la gestion des données.
Installez-les via :

bash
Copy code
pip install transformers spacy matplotlib seaborn scikit-learn pandas
Fichiers Importants
reviews.jsonl : Contient les avis clients.
preprocessed_reviews.json : Avis nettoyés, prêts pour le clustering.
Rapport_Projet.pdf : Rapport final contenant une analyse complète.
Améliorations Futures
Intégrer des techniques avancées comme LDA ou BERTopic pour le topic modeling.
Utiliser des modèles plus spécifiques pour l'analyse des sentiments (par exemple, DistilBERT).
Déployer le pipeline complet dans une application web pour une visualisation interactive.
Auteur
Nom : HAMZAOUI Assia, ZIDELMAL Hamid
Contact : assiahamzaoui72@gmail.com







