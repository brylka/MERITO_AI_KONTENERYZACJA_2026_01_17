from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib


print("Trenuję model...")
iris = load_iris()
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(iris.data, iris.target)

joblib.dump(model, 'model.pkl')
print("Model zapisany do pliku model.pkl!")