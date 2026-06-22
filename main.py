from src.models import get_model
from src.models import load_data

model = get_model("lgbm")
X_train, X_test, y_train, y_test = load_data()
model.fit(X_train, y_train)