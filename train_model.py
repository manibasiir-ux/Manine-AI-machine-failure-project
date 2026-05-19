import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("machine_failure_dataset.csv")

feature_cols = [
    'Temperature',
    'Vibration',
    'Power_Usage',
    'Humidity',
    'Machine_Type']

target_col = 'Failure_Risk'

X = df[feature_cols]
y = df[target_col]

# =========================
# PREPROCESSING
# =========================

numerical_features = [
    'Temperature',
    'Vibration',
    'Power_Usage',
    'Humidity']

categorical_features = ['Machine_Type']

numeric_transformer = 'passthrough'
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numerical_features),
                                               ('cat', categorical_transformer, categorical_features)])

# =========================
# SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=4,
    stratify=y)

# =========================
# DECISION TREE MODEL
# =========================

dt_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=8,
        class_weight='balanced',
        random_state=42))])

dt_pipeline.fit(X_train, y_train)

# =========================
# RANDOM FOREST MODEL
# =========================

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42))])

rf_pipeline.fit(X_train, y_train)

# =========================
# SAVE MODELS
# =========================

joblib.dump(
    dt_pipeline,'decisiontree_machine_failure_model.pkl')

joblib.dump(
    rf_pipeline,'random_forest_explainer.pkl')

print("Models saved successfully.")

