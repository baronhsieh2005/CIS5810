from transformers import pipeline
clf = pipeline("image-classification", model="./vit_squat_model")
print(clf("runs\pose\predict\ezgif-21-020.jpg", top_k=None))