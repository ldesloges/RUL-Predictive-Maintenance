# Variables
MSG = "Mise à jour automatique du modèle et du dashboard"

# Commande complète pour sauvegarder et envoyer sur GitHub
push:
	@echo "🚀 Préparation de l'envoi vers GitHub..."
	git add .
	git commit -m "gr"
	git push origin main
	@echo "✅ Terminé ! Ton code est sur https://github.com/ldesloges/RUL-Predictive-Maintenance"

# Commande pour installer les dépendances (utile pour le déploiement)
install:
	pip install -r requirements.txt

# Commande pour lancer le dashboard en local
run:
	streamlit run app.py