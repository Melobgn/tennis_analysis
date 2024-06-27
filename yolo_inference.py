from ultralytics import YOLO
import os

# Charger le modèle
model = YOLO('yolov8n.pt')

# Définir le répertoire de sauvegarde
project_dir = '/home/utilisateur/Documents/Computer vision/tennis_analysis/runs'
output_name = 'predict'

# Créer le répertoire si nécessaire
os.makedirs(os.path.join(project_dir, output_name), exist_ok=True)

# Effectuer la prédiction et sauvegarder les résultats
results = model.predict(
    source='/home/utilisateur/Documents/Computer vision/tennis_analysis/input_videos/image.png',
    save=True,
    project=project_dir,
    name=output_name
)

# Afficher où les résultats ont été sauvegardés
print(f"Les résultats ont été sauvegardés dans: {os.path.join(project_dir, output_name)}")

