import os
import pickle
import numpy as np
import face_recognition

# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FACE_DB = os.path.join(BASE_DIR, "face_database")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.pkl")

# ==========================================================
# Generate embeddings
# ==========================================================

database = {}

print("=" * 60)
print("Generating Face Embeddings...")
print("=" * 60)

for person_name in sorted(os.listdir(FACE_DB)):

    person_folder = os.path.join(FACE_DB, person_name)

    if not os.path.isdir(person_folder):
        continue

    person_embeddings = []

    print(f"\nProcessing: {person_name}")

    for image_name in sorted(os.listdir(person_folder)):

        image_path = os.path.join(person_folder, image_name)

        if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:

            image = face_recognition.load_image_file(image_path)

            encodings = face_recognition.face_encodings(image)

            if len(encodings) == 0:
                print(f"  No face found: {image_name}")
                continue

            person_embeddings.append(encodings[0])

            print(f"  OK: {image_name}")

        except Exception as e:
            print(f"  ERROR: {image_name}")
            print(e)

    if len(person_embeddings) == 0:
        print(f"  No usable images for {person_name}")
        continue

    average_embedding = np.mean(person_embeddings, axis=0)

    database[person_name] = average_embedding.tolist()

    print(f"  Saved {len(person_embeddings)} embeddings")

# ==========================================================
# Save
# ==========================================================

with open(EMBEDDINGS_PATH, "wb") as f:
    pickle.dump(database, f)

print("\n")
print("=" * 60)
print("Done!")
print(f"People registered: {len(database)}")
print("Saved to:", EMBEDDINGS_PATH)
print("=" * 60)