

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from lable_encode import label_encoder
from extract import X_train, y_train, X_val, y_val
import joblib

model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Validate the model
y_val_pred = model.predict(X_val)
print("Validation Results:\n", classification_report(y_val, y_val_pred, target_names=label_encoder.classes_))
model_filename = 'asd_classification_model_svm.pkl'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

