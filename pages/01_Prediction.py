import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

df_var_imp_detect_U19 = pd.read_csv("df_var_importantes_detect_U19.csv",sep=",")
df_var_imp_detect_U19 = df_var_imp_detect_U19.drop(columns=['Unnamed: 0'])
df_var_imp_detect_U20 = pd.read_csv("df_var_importantes_detect_U20.csv",sep=",")
df_var_imp_detect_U20 = df_var_imp_detect_U20.drop(columns=['Unnamed: 0'])
df_var_imp_U16_U19 = pd.read_csv("df_var_importantes_U16_U19.csv",sep=",")
df_var_imp_U16_U19 = df_var_imp_U16_U19.drop(columns=['Unnamed: 0'])
df_var_imp_U18_U20 = pd.read_csv("df_var_importantes_U18_U20.csv",sep=",")
df_var_imp_U18_U20 = df_var_imp_U18_U20.drop(columns=['Unnamed: 0'])
df_var_imp_U19_U21 = pd.read_csv("df_var_importantes_U19_U21.csv",sep=",")
df_var_imp_U19_U21 = df_var_imp_U19_U21.drop(columns=['Unnamed: 0'])

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1200px;
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre de l'application
st.subheader("Prédiction du parcours des joueurs")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Choisissez le fichier CSV correspondant aux données imputées", type="csv")


# Charger les modèles
models = {
   "Aucune_U16": 'model_detect_U16.joblib',
   "Aucune_U17" : 'model_detect_U17.joblib',
   "Aucune_U18": 'model_detect_U18.joblib',
   "Aucune_U19": 'model_detect_U19.joblib',
   "Aucune_U20": 'model_detect_U20.joblib',
   "Aucune_U21": 'model_detect_U21.joblib',
   "Aucune_A": 'model_detect_A.joblib',
   "U16_U17": 'model_U16_U17.joblib',
   "U16_U18" : 'model_U16_U18.joblib',
   "U16_U19": 'model_U16_U19.joblib',
   "U16_U20": 'model_U16_U20.joblib',
   "U17_U18": 'model_U17_U18.joblib',
   "U17_U19": 'model_U17_U19.joblib',
   "U17_U20": 'model_U17_U20.joblib',
   "U18_U19": 'model_U18_U19.joblib',
   "U18_U20": 'model_U18_U20.joblib',
   "U19_U20": 'model_U19_U20.joblib',
   "U19_U21": 'model_U19_U21.joblib',
   "U20_U21": 'model_U20_U21.joblib',
   "U21_A" : 'model_U21_edfA.joblib'
}


# Spécifier les colonnes nécessaires pour chaque modèle
model_columns = {
       "Aucune_U16": ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
        "NB_PALMARES_AVANT_16" ,  "exp_avant_16" ,  "Anthropometrie" ,
        "Detente_horizontale" ,  "Vitesse_lineaire" ,  "Vivacite" ,  "Jeux_reduits" ,
        "Match" ,  "Conduite_de_balle" ,  "Gardiens_de_but" ,
        "Jonglerie_en_mouvement" ,  "Vitesse_max_aerobie" ,  
        "NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16", "NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16", 
          "NB_CLUB_AVANT_16" ,  "TOP_50_AVANT_16" , "TOP_50_150_AVANT_16" ,  "top_restant_AVANT_16"],

      "Aucune_U17" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_17', 'exp_avant_17', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_17',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_17', 'NB_CLUB_AVANT_17',
       'TOP_50_AVANT_17', 'TOP_50_150_AVANT_17', 'top_restant_AVANT_17',
       'France U16', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION'],
        

      "Aucune_U18" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_18', 'exp_avant_18', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_18',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_18', 'NB_CLUB_AVANT_18',
       'TOP_50_AVANT_18', 'TOP_50_150_AVANT_18', 'top_restant_AVANT_18',
       'France U16', 'France U17', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE',
       'U16_NB_MIN_JOUE', 'U16_TITULARISATION', 'U17_NB_BUT',
       'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE', 'U17_TITULARISATION'],

      "Aucune_U19" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_19', 'exp_avant_19', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_19',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_19', 'NB_CLUB_AVANT_19',
       'TOP_50_AVANT_19', 'TOP_50_150_AVANT_19', 'top_restant_AVANT_19',
       'France U16', 'France U17', 'France U18', 'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION'],

      "Aucune_U20" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_20', 'exp_avant_20', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_20',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_20', 'NB_CLUB_AVANT_20',
       'TOP_50_AVANT_20', 'TOP_50_150_AVANT_20', 'top_restant_AVANT_20',
       'France U16', 'France U17', 'France U18', 'France U19', 'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION', 'U19_NB_BUT',
       'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE', 'U19_TITULARISATION'],

      "Aucune_U21" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_21', 'exp_avant_21', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_21',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_21', 'NB_CLUB_AVANT_21',
       'TOP_50_AVANT_21', 'TOP_50_150_AVANT_21', 'top_restant_AVANT_21',
       'France U16', 'France U17', 'France U18', 'France U19', 'France U20',
       'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION', 'U17_NB_BUT', 'U17_NB_MATCH_JOUE',
       'U17_NB_MIN_JOUE', 'U17_TITULARISATION', 'U18_NB_BUT',
       'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE', 'U18_TITULARISATION',
       'U19_NB_BUT', 'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE',
       'U19_TITULARISATION', 'U20_NB_BUT', 'U20_NB_MATCH_JOUE',
       'U20_NB_MIN_JOUE', 'U20_TITULARISATION'],
      
      "U16": ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_16', 'exp_avant_16', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16', 'NB_CLUB_AVANT_16',
       'TOP_50_AVANT_16', 'TOP_50_150_AVANT_16', 'top_restant_AVANT_16',
       'France U16', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION'],

      "U17" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_17', 'exp_avant_17', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_17',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_17', 'NB_CLUB_AVANT_17',
       'TOP_50_AVANT_17', 'TOP_50_150_AVANT_17', 'top_restant_AVANT_17',
       'France U16', 'France U17', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE',
       'U16_NB_MIN_JOUE', 'U16_TITULARISATION', 'U17_NB_BUT',
       'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE', 'U17_TITULARISATION'],

      "U18" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_18', 'exp_avant_18', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_18',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_18', 'NB_CLUB_AVANT_18',
       'TOP_50_AVANT_18', 'TOP_50_150_AVANT_18', 'top_restant_AVANT_18',
       'France U16', 'France U17', 'France U18', 'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION'],

      "U19" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_19', 'exp_avant_19', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_19',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_19', 'NB_CLUB_AVANT_19',
       'TOP_50_AVANT_19', 'TOP_50_150_AVANT_19', 'top_restant_AVANT_19',
       'France U16', 'France U17', 'France U18', 'France U19', 'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION', 'U19_NB_BUT',
       'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE', 'U19_TITULARISATION'],

        "U20" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_20', 'exp_avant_20', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_20',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_20', 'NB_CLUB_AVANT_20',
       'TOP_50_AVANT_20', 'TOP_50_150_AVANT_20', 'top_restant_AVANT_20',
       'France U16', 'France U17', 'France U18', 'France U19', 'France U20',
       'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION', 'U17_NB_BUT', 'U17_NB_MATCH_JOUE',
       'U17_NB_MIN_JOUE', 'U17_TITULARISATION', 'U18_NB_BUT',
       'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE', 'U18_TITULARISATION',
       'U19_NB_BUT', 'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE',
       'U19_TITULARISATION', 'U20_NB_BUT', 'U20_NB_MATCH_JOUE',
       'U20_NB_MIN_JOUE', 'U20_TITULARISATION'],

         "U21" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES_AVANT_21', 'exp_avant_21', 'Anthropometrie',
       'Detente_horizontale', 'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits',
       'Match', 'Conduite_de_balle', 'Gardiens_de_but',
       'Jonglerie_en_mouvement', 'Vitesse_max_aerobie',
       'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_21',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_21', 'NB_CLUB_AVANT_21',
       'TOP_50_AVANT_21', 'TOP_50_150_AVANT_21', 'top_restant_AVANT_21',
       'France U16', 'France U17', 'France U18', 'France U19', 'France U20',
       'France U21', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION', 'U17_NB_BUT', 'U17_NB_MATCH_JOUE',
       'U17_NB_MIN_JOUE', 'U17_TITULARISATION', 'U18_NB_BUT',
       'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE', 'U18_TITULARISATION',
       'U19_NB_BUT', 'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE',
       'U19_TITULARISATION', 'U20_NB_BUT', 'U20_NB_MATCH_JOUE',
       'U20_NB_MIN_JOUE', 'U20_TITULARISATION', 'U21_NB_BUT',
       'U21_NB_MATCH_JOUE', 'U21_NB_MIN_JOUE', 'U21_TITULARISATION'],

       "Aucune_A" : ['POSTE', 'MOIS_NAISSANCE', 'NATIONALITE', 'CF_AGREE', 'POLE_ESPOIR',
       'NB_PALMARES', 'exp', 'Anthropometrie', 'Detente_horizontale',
       'Vitesse_lineaire', 'Vivacite', 'Jeux_reduits', 'Match',
       'Conduite_de_balle', 'Gardiens_de_but', 'Jonglerie_en_mouvement',
       'Vitesse_max_aerobie', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON', 'NB_CLUB', 'TOP_50', 'TOP_50_150',
       'top_restant', 'France U16', 'France U17', 'France U18', 'France U19',
       'France U20', 'France U21', 'U16_NB_BUT', 'U16_NB_MATCH_JOUE',
       'U16_NB_MIN_JOUE', 'U16_TITULARISATION', 'U17_NB_BUT',
       'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE', 'U17_TITULARISATION',
       'U18_NB_BUT', 'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE',
       'U18_TITULARISATION', 'U19_NB_BUT', 'U19_NB_MATCH_JOUE',
       'U19_NB_MIN_JOUE', 'U19_TITULARISATION', 'U20_NB_BUT',
       'U20_NB_MATCH_JOUE', 'U20_NB_MIN_JOUE', 'U20_TITULARISATION',
       'U21_NB_BUT', 'U21_NB_MATCH_JOUE', 'U21_NB_MIN_JOUE',
       'U21_TITULARISATION']
}




# Fonction pour effectuer la prédiction
def predict(model, donnees):
    prediction = model.predict(donnees)
    prediction_proba = model.predict_proba(donnees)
    return prediction, prediction_proba


if uploaded_file is not None:
    # Lecture du fichier CSV
    data = pd.read_csv(uploaded_file, encoding='utf-8',sep=",")
    data['NOM_COMPLET'] = data['NOM']+ ' ' + data['PRENOM']
    
    # Sélection du joueur
    player_name = st.sidebar.selectbox('Sélectionnez un joueur', data['NOM_COMPLET'].unique())

      # Sélection de la catégorie d'âge ACTUELLE
    age_cat_actuel = st.sidebar.selectbox('Sélectionnez la sélection dans laquelle est le joueur', ['Aucune', 'U16', 'U17', 'U18', 'U19','U20','U21'])

      # Sélection de la catégorie d'âge de prédiction
    if(age_cat_actuel=="Aucune"):
        age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U16', 'U17', 'U18', 'U19','U20','U21',"Equipe de France A"])
    elif(age_cat_actuel=="U16"):
           age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U17', 'U18', 'U19','U20'])
    elif(age_cat_actuel=="U17"):
             age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U18', 'U19','U20'])
    elif(age_cat_actuel=="U18"):
               age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U19','U20'])
    elif(age_cat_actuel=="U19"):
               age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U20',"U21"])
    elif(age_cat_actuel=="U20"):
               age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['U21'])
    else:
               age_cat_pred = st.sidebar.selectbox('Sélectionnez la catégorie d\'âge dans laquelle vous voulez connaitre la prédiction', ['Equipe de France A'])   

# Bouton de prédiction
if st.sidebar.button('Predict'):
   
   # Selections des variables nécessaires pour la prédiction
   if(age_cat_actuel=="Aucune"):
         if(age_cat_pred=="Equipe de France A"):
            age_cat_pred="A"
         age_model = age_cat_actuel + "_" + age_cat_pred
         model_choisis = models[age_model]
         features_model = model_columns[age_model]
   else:
          if(age_cat_pred=="Equipe de France A"):
            age_cat_pred="A"
          age = age_cat_actuel + "_" + age_cat_pred
          model_choisis = models[age]
          features_model = model_columns[age_cat_actuel]
          
  
   
   # Concatener la valeur de age_cat_actuel et age_cat_pred pour obtenir le nom du modèle (sep = "_")
   
   # Choix du model
   
   model = joblib.load(filename = model_choisis)
      
   # Extraction des informations du joueur
   player_data = data[data['NOM_COMPLET'] == player_name].iloc[0]
   player_data_pred = player_data[features_model]
   player_data_pred = player_data_pred.values.reshape(1, -1)

   # Transformer le numpy array en DataFrame
   player_data_pred = pd.DataFrame(player_data_pred, columns=features_model)

   # Transformation des données
   player_data_pred.iloc[:, 19:23] = player_data_pred.iloc[:, 19:23].astype('object')


   # On récupère la valeur de 
   prediction, prediction_proba = predict(model, player_data_pred)
   st.write('<u>Résultat de la prédiction:</u> classe ', str(prediction[0]), unsafe_allow_html=True)
   if prediction[0] == 1:
      st.write(f'Ce résultat signifie que {player_name} **a ses chances** de jouer en sélection nationale')
   else:
      st.write(f'Ce résultat signifie que {player_name} n\' a pour l\'instant **pas ses chances** de jouer en sélection nationale {str(age_cat_pred)}')
   
   st.write(f'D\'après le modèle de prédiction, {player_name} a **{round(prediction_proba[:,1][0]*100,2)}%** de chance de jouer en équipe de France {str(age_cat_pred)}')
   
   
   ##############################################################################
   # Variable d'importance
   
   feature_names = model.named_steps['preprocessor'].get_feature_names_out()
   model_name_map = {
    'clf_knn': KNeighborsClassifier,
    'xgbclassifier': XGBClassifier,
    'classifier': RandomForestClassifier,  
    'logistic_regression': LogisticRegression,     
  }

# Parcourir les étapes et vérifier leur type
   for step, autre in model.named_steps.items():
    # Vérifier si l'objet correspond au type attendu pour le nom d'étape
      if step in model_name_map.keys():
         step_model = step
   
   if step_model != 'clf_knn':
        st.subheader('Variables d\'importance')
      # Récupérer l'importance des variables
        # st.write(step_model)
        if step_model == 'xgbclassifier' or step_model == 'classifier':
              importances = model.named_steps[step_model].feature_importances_
              feature_importance_df = pd.DataFrame({
                  'feature': feature_names,
                  'importance': importances
              })
        elif step_model == 'logistic_regression':
              importances = model.named_steps[step_model].coef_[0]
              feature_importance_df = pd.DataFrame({
                  'feature': feature_names,
                  'importance': importances
              })
               
      
        # Créer un DataFrame pour associer les importances aux noms de variables
        
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        feature_importance_df['feature'] = feature_importance_df['feature'].str.replace('cat__', '')
        feature_importance_df['feature'] = feature_importance_df['feature'].str.replace('num__', '')
        top_5_features = feature_importance_df.head(5)
        # si top_5_features["feature"] commence par 'TOP' alors on affiche le nom complet de la variable
          # sinon on affiche la variable sans le préfixe
        if(top_5_features["feature"].str.startswith('TOP').any()):
            top_5_features["feature"] = top_5_features["feature"].str.replace('TOP_50_AVANT_16_inconnu','TOP_50')
            top_5_features["feature"] = top_5_features["feature"].str.replace('TOP_50_150_AVANT_16_inconnu','TOP_50_150')
            top_5_features["feature"] = top_5_features["feature"].str.replace('top_restant_AVANT_16_inconnu','top_restant')
        
        # top_5_features_names = top_5_features[['feature']].rename(columns={'feature': 'Variables importantes'})
        top_5_features_names = top_5_features.rename(columns={'feature': 'Variables importantes'})

        # Réinitialiser l'index pour supprimer les numéros de ligne
        top_5_features_names_reset = top_5_features_names.reset_index(drop=True)

        st.write(top_5_features_names_reset)
        # Convertir en HTML avec style pour changer la couleur de fond des en-têtes
        # table_html = top_5_features_names_reset.to_html(index=False)

        # Styliser les en-têtes
        # styled_table = """
        # <style>
        # thead th {
        #     background-color: #fff1ba; /* Changer ici la couleur de fond des en-têtes */
        #     color: black; /* Couleur du texte des en-têtes */
        #     text-align: left;
        # }
        # </style>
        # """

        # # Afficher le tableau stylisé
        # st.markdown(styled_table + top_5_features_names_reset, unsafe_allow_html=True)
   else:
        st.subheader('Variables d\'importance')
        if(age_cat_actuel=="Aucune"):
            age = age_cat_actuel + "_" + age_cat_pred
        else:
                if(age_cat_pred=="Equipe de France A"):
                  age_cat_pred="A"
                age = age_cat_actuel + "_" + age_cat_pred

        if age == 'Aucune_U19':
              st.write(df_var_imp_detect_U19)
        elif age == 'Aucune_U20':
              st.write(df_var_imp_detect_U20)
        elif age == 'U16_U19':
              st.write(df_var_imp_U16_U19)
        elif age == 'U18_U20':
              st.write(df_var_imp_U18_U20)
        elif age == 'U19_U21':
              st.write(df_var_imp_U19_U21)
