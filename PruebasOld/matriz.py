import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

val_gen.reset()
preds = model.predict(val_gen)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)