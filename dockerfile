# Utilise une image Python légère
FROM python:3.11-slim

# Évite les logs tronqués
ENV PYTHONUNBUFFERED True

# Copie le code dans le conteneur
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Lance l'application avec Gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
