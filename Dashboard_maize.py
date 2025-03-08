import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Configurer Streamlit
st.set_page_config(
    page_title="Dashboard de Prédiction du Maïs",
    page_icon="🌽",
    layout="wide",
)

# Bannière
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #228B22;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 20px;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <div class="main-title">🌽 Dashboard de Prédiction du Maïs 🌽</div>
    <div class="sub-title">Analyse, Visualisation et Comparaison des Modèles</div>
    """,
    unsafe_allow_html=True,
)

# Barre latérale
st.sidebar.header("🔄 Configuration")
uploaded_file = st.sidebar.file_uploader(
    "Chargez un fichier Excel (avec une feuille 'Tmnt1')", type=["xlsx"]
)
st.sidebar.markdown("Configurez les options pour l'analyse et la visualisation.")

# Chargement des données
if uploaded_file:
    try:
        tmnt1_data = pd.read_excel(uploaded_file, sheet_name="Tmnt1")
        st.sidebar.success("Fichier chargé avec succès !")
    except Exception as e:
        st.sidebar.error(f"Erreur : {e}")
        st.stop()

    # Préparation des données
    selected_columns = ["canopy Cover", "Soil Water Deficit (mm)", "DAS"]

    def prepare_data(data, selected_columns):
        cleaned_data = data[selected_columns].dropna()
        cleaned_data = cleaned_data.apply(pd.to_numeric, errors="coerce").dropna()
        return cleaned_data

    data = prepare_data(tmnt1_data, selected_columns)

    # Mise en page : deux colonnes
    col1, col2 = st.columns([2, 1])

    # Affichage des données
    with col1:
        st.subheader("🔍 Aperçu des Données")
        st.dataframe(data, use_container_width=True)

    # Statistiques de base
    with col2:
        st.subheader("📊 Statistiques Descriptives")
        st.write(data.describe())

    # Visualisation des données
    st.markdown("### 📈 Visualisation des Données")
    st.markdown("**Analyse des distributions des variables sélectionnées :**")

    # Distribution des colonnes
    for column in data.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(data[column], kde=True, bins=20, color="green", ax=ax)
        ax.set_title(f"Distribution de {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Fréquence")
        st.pyplot(fig)

    # Matrice de corrélation
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap="YlGnBu", fmt=".2f", ax=ax)
    ax.set_title("Matrice de Corrélation")
    st.pyplot(fig)

    # Modélisation et Prédiction
    def model_and_predict(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        results = {}

        # Régression linéaire
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred_linear = linear_model.predict(X_test)
        results["linear"] = {
            "model": linear_model,
            "mse": mean_squared_error(y_test, y_pred_linear),
            "r2": r2_score(y_test, y_pred_linear),
            "predictions": y_pred_linear,
        }

        # Arbre de décision
        tree_model = DecisionTreeRegressor(random_state=42)
        tree_model.fit(X_train, y_train)
        y_pred_tree = tree_model.predict(X_test)
        results["tree"] = {
            "model": tree_model,
            "mse": mean_squared_error(y_test, y_pred_tree),
            "r2": r2_score(y_test, y_pred_tree),
            "predictions": y_pred_tree,
        }

        # Forêt aléatoire
        forest_model = RandomForestRegressor(random_state=42, n_estimators=100)
        forest_model.fit(X_train, y_train)
        y_pred_forest = forest_model.predict(X_test)
        results["forest"] = {
            "model": forest_model,
            "mse": mean_squared_error(y_test, y_pred_forest),
            "r2": r2_score(y_test, y_pred_forest),
            "predictions": y_pred_forest,
        }

        return results, y_test

    X = data[["Soil Water Deficit (mm)", "DAS"]]
    y = data["canopy Cover"]
    results, y_test = model_and_predict(X, y)

    # Visualisation des modèles
    st.markdown("### 🌟 Visualisation des Prédictions par Modèle")
    tab1, tab2, tab3 = st.tabs(
        ["Régression Linéaire", "Arbre de Décision", "Forêt Aléatoire"]
    )

    def plot_model_predictions(model_name, y_test, predictions):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test.values, label="Données réelles", marker="o", color="blue")
        ax.plot(
            predictions,
            label=f"Prédictions {model_name.capitalize()}",
            linestyle="--",
            color="orange",
        )
        ax.set_title(f"Prédictions pour {model_name.capitalize()}")
        ax.set_xlabel("Index")
        ax.set_ylabel("Canopy Cover")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

    with tab1:
        plot_model_predictions("linear", y_test, results["linear"]["predictions"])

    with tab2:
        plot_model_predictions("tree", y_test, results["tree"]["predictions"])

    with tab3:
        plot_model_predictions("forest", y_test, results["forest"]["predictions"])

    # Comparaison des modèles
    st.markdown("### 🔄 Comparaison des Modèles")
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, result in results.items():
        ax.plot(result["predictions"], label=f"Prédictions {model_name.capitalize()}")
    ax.plot(y_test.values, label="Données Réelles", linestyle="--", color="black")
    ax.set_title("Comparaison des Prédictions des Modèles")
    ax.set_xlabel("Index")
    ax.set_ylabel("Canopy Cover")
    ax.legend()
    ax.grid()
    st.pyplot(fig)

    # Résultats finaux
    best_model = min(results, key=lambda x: results[x]["mse"])
    st.success(f"**Le meilleur modèle est : {best_model.capitalize()}**")
    st.markdown(
        f"""
        - **Erreur Quadratique Moyenne (MSE) :** {results[best_model]['mse']:.3f}
        - **R² :** {results[best_model]['r2']:.3f}
        """
    )
else:
    st.warning("Veuillez charger un fichier Excel pour commencer.")
