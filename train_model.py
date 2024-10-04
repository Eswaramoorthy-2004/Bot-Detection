import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate synthetic data
# Features: [average movementX, average movementY]
# Labels: 0 for human, 1 for bot
X = np.random.rand(1000, 2)  # Random data for demonstration
y = np.random.randint(0, 2, 1000)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model to file
with open('mouse_movement_model.pkl', 'wb') as f:
    pickle.dump(model, f)
