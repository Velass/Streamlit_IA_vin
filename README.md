# 🍷 ClassiVin – Analyse et Machine Learning sur un dataset de vins

Application interactive développée avec **Python** et **Streamlit**, permettant d’explorer un dataset de vins, de préparer les données et d’entraîner un modèle de **Machine Learning** pour prédire la catégorie d’un vin.

L’application présente de manière structurée les différentes étapes d’un projet de **data science**, depuis l’exploration des données jusqu’à la génération de prédictions, à travers une interface web simple et interactive.

---

# Objectif du projet

Ce projet illustre les principales étapes d’un projet **Machine Learning** :

* exploration d’un dataset
* analyse statistique des données
* préparation et transformation des variables
* entraînement d’un modèle de Machine Learning
* évaluation des performances
* génération de prédictions

L’utilisation de **Streamlit** permet de rendre ces étapes **interactives et accessibles via une interface web**.

---

# Technologies utilisées

* Python 3.12
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

# Structure du projet

Le projet est organisé sous la forme d’une **application Streamlit multi-pages**.

```text
ClassiVin
│
├── Streamlit
│   ├── Accueil.py
│   └── pages
│
├── data
│
└── requirements.txt
```

Chaque page de l’application correspond à une étape du pipeline de data science.

---

# Fonctionnalités principales

## Exploration des données

* affichage du dataset
* statistiques descriptives
* visualisation des données

## Préparation des données

* nettoyage du dataset
* transformation des variables
* préparation des données pour le Machine Learning

## Machine Learning

* entraînement d’un modèle
* génération de prédictions
* évaluation des performances

---

# Pipeline Machine Learning

Le projet suit les étapes classiques d’un projet de data science :

Dataset
↓
Exploration des données
↓
Préparation des données
↓
Entraînement du modèle
↓
Évaluation
↓
Prédictions

---

# Installation

Prérequis
Avant de lancer le projet, il est nécessaire d’avoir installé :
•	Python 3.12
•	pip
Le projet a été développé et testé avec Python 3.12.

## Cloner le projet

```bash
git clone https://github.com/Velass/Streamlit_IA_vin.git
cd Streamlit_IA_vin
```

---

## Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# Lancer l'application

Pour lancer l’application Streamlit, depuis la racine du projet, exécuter la commande suivante :

```bash
streamlit run Streamlit/Accueil.py
```

L’application sera ensuite accessible à l’adresse suivante :

```
http://localhost:8501
```

---

# Travail collaboratif

Ce projet a été réalisé **en collaboration avec deux autres personnes** dans le cadre d’un projet de data science.

Les tâches principales ont consisté à :

* analyser et comprendre le dataset
* préparer les données pour le Machine Learning
* entraîner et évaluer un modèle
* développer une interface interactive avec Streamlit

---

# Compétences mises en pratique

Ce projet met en pratique :

* la manipulation de données avec **Pandas**
* la création d’un **pipeline Machine Learning**
* l’utilisation de **Scikit-learn**
* le développement d’une application interactive avec **Streamlit**
* le travail **collaboratif sur un projet data**

---

# Auteur

Projet réalisé par **Velass** en collaboration avec deux autres personnes.

---

# Note

Ce README a été **rédigé avec l’aide d’un assistant IA (ChatGPT)** puis ajusté pour documenter le projet.
