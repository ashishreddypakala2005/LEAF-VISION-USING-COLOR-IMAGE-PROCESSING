from pathlib import Path
from src.dataset_utils import scan_dataset
root=Path(__file__).parent/"dataset";stats=scan_dataset(root)
print("LeafVision dataset check");print("Folder:",root);print("Images:",stats["image_count"]);print("Class folders:",stats["class_count"]);print("Approx size MB:",round(stats["size_mb"],1))
for name,count in stats["classes"]: print(f"{name}: {count}")
if not stats["image_count"]: print("Add PlantVillage class folders inside dataset/.")
