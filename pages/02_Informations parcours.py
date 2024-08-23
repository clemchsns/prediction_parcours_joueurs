import streamlit as st
import pandas as pd
import joblib
import numpy as np

pd.set_option('display.max_columns', None)  # Afficher toutes les colonnes
pd.set_option('display.width', None)    

bd_app_model = pd.read_csv("bd_app_modelisation.csv",sep=",")

st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 100%;
        padding-left: 5%;
        padding-right: 5%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Titre de l'application
st.subheader("Informations sur le parcours des joueurs")

# Chargement du fichier CSV
st.write("Veuillez charger le fichier \'data.csv\'")
uploaded_data = st.file_uploader("Chargez le fichier", type="csv")

cat_age_actuelle_columns = {
    "Aucune" : ['SAISONS_KPI_AVANT_16', 'TYPES_KPI_AVANT_16', 'VALUES_AVANT_16',
                'SAISONS_EQ_AVANT_16', 'EQUIPES_AVANT_16',
        'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16'],
    "U16": ['SAISONS_KPI_AVANT_16', 'TYPES_KPI_AVANT_16', 'VALUES_AVANT_16',
            'SAISONS_EQ_AVANT_16', 'EQUIPES_AVANT_16',
            'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16', 
       'U16_NB_BUT','U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE','U16_TITULARISATION'], 
    "U17": ['SAISONS_KPI_AVANT_17', 'TYPES_KPI_AVANT_17', 'VALUES_AVANT_17',
            'SAISONS_EQ_AVANT_18', 'EQUIPES_AVANT_18',
            'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_17', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_17', 
       'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE', 'U17_TITULARISATION',
       'France U16'],
    "U18": ['SAISONS_KPI_AVANT_18', 'TYPES_KPI_AVANT_18', 'VALUES_AVANT_18',
            'SAISONS_EQ_AVANT_18', 'EQUIPES_AVANT_18',
            'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_18', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_18', 
        'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION', 'U17_NB_BUT', 'U17_NB_MATCH_JOUE',
       'U17_NB_MIN_JOUE', 'U17_TITULARISATION', 'U18_NB_BUT',
        'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE', 'U18_TITULARISATION',
        'France U16', 'France U17'],
    "U19": ['SAISONS_KPI_AVANT_19', 'TYPES_KPI_AVANT_19', 'VALUES_AVANT_19',
            'SAISONS_EQ_AVANT_19', 'EQUIPES_AVANT_19',
             'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_19', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_19',
             'U16_NB_BUT', 'U16_NB_MATCH_JOUE',
       'U16_NB_MIN_JOUE', 'U16_TITULARISATION', 'U17_NB_BUT',
       'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE', 'U17_TITULARISATION',
       'U18_NB_BUT', 'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE',
       'U18_TITULARISATION','U19_NB_BUT', 'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE', 'U19_TITULARISATION',
       'France U16','France U17','France U18'
       ],
    "U20": ['SAISONS_KPI_AVANT_20', 'TYPES_KPI_AVANT_20', 'VALUES_AVANT_20',
            'NB_BUT_AVANT_20',
            'SAISONS_EQ_AVANT_20', 'EQUIPES_AVANT_20',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_20', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_20',
       'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION', 'U19_NB_BUT',
       'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE', 'U19_TITULARISATION',
       'U20_NB_BUT', 'U20_NB_MATCH_JOUE', 'U20_NB_MIN_JOUE','U20_TITULARISATION',
         'France U16','France U17','France U18', 'France U19'],
    "U21": ['SAISONS_KPI_AVANT_21', 'TYPES_KPI_AVANT_21', 'VALUES_AVANT_21',
            'NB_BUT_AVANT_21',
            'SAISONS_EQ_AVANT_16', 'EQUIPES_AVANT_16',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_21', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_21',
         'U16_NB_BUT',
       'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE', 'U16_TITULARISATION',
       'U17_NB_BUT', 'U17_NB_MATCH_JOUE', 'U17_NB_MIN_JOUE',
       'U17_TITULARISATION', 'U18_NB_BUT', 'U18_NB_MATCH_JOUE',
       'U18_NB_MIN_JOUE', 'U18_TITULARISATION', 'U19_NB_BUT',
       'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE', 'U19_TITULARISATION',
       'U20_NB_BUT', 'U20_NB_MATCH_JOUE', 'U20_NB_MIN_JOUE',
       'U20_TITULARISATION',
       'U21_NB_BUT','U21_NB_MATCH_JOUE', 'U21_NB_MIN_JOUE', 'U21_TITULARISATION',
       'France U16', 'France U17','France U18', 'France U19', 'France U20'],
    "Equipe de France A": [
       'SAISONS_KPI_TOTAL', 'TYPES_KPI_TOTAL', 'VALUES_TOTAL',
       'SAISONS_EQ_TOTAL', 'EQUIPES_TOTAL',
       'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_TOTAL', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_TOTAL',
       'U16_NB_BUT', 'U16_NB_MATCH_JOUE', 'U16_NB_MIN_JOUE',
       'U16_TITULARISATION', 'U17_NB_BUT', 'U17_NB_MATCH_JOUE',
       'U17_NB_MIN_JOUE', 'U17_TITULARISATION', 'U18_NB_BUT',
       'U18_NB_MATCH_JOUE', 'U18_NB_MIN_JOUE', 'U18_TITULARISATION',
       'U19_NB_BUT', 'U19_NB_MATCH_JOUE', 'U19_NB_MIN_JOUE',
       'U19_TITULARISATION', 'U20_NB_BUT', 'U20_NB_MATCH_JOUE',
       'U20_NB_MIN_JOUE', 'U20_TITULARISATION', 'U21_NB_BUT',
       'U21_NB_MATCH_JOUE', 'U21_NB_MIN_JOUE', 'U21_TITULARISATION',
       'France_NB_BUT', 'France_NB_MATCH_JOUE', 'France_NB_MIN_JOUE',
       'France_TITULARISATION',
       'France U16', 'France U17', 'France U21', 'France U18', 'France U19', 'France U20']
}



# Choix du joueurs

if uploaded_data is not None:
    df = pd.read_csv(uploaded_data)
    
    df['NOM_COMPLET'] = df['NOM']+ ' ' + df['PRENOM']
    joueur = st.sidebar.selectbox("Choisissez le nom du joueur", df['NOM_COMPLET'])

    # categorie = st.sidebar.selectbox("Choisissez la sélection dans laquelle le joueur est actuellement", ['Aucune', 'U16', 'U17', 'U18', 'U19', 'U20', 'U21'])
    # st.write(f"Le joueur est actuellement en : {categorie}")

    # colonnes = cat_age_actuelle_columns[categorie]
    df_player = df[df['NOM_COMPLET'] == joueur]
    # df_player = df_player[colonnes]
    # st.write(df_player)
    
    # features = cat_columns[categorie]
    # df_player = df_player[features]


    ############################################
    st.subheader("Le profil du joueur")
    # st.write(df_player)
    df_infos_joueur = df_player[['NOM', 'PRENOM', 'DATE_NAISSANCE', 'POSTE', 'CF_AGREE', 'POLE_ESPOIR']]
    if df_infos_joueur['POSTE'].isna().all():
        df_infos_joueur['POSTE'] = 'Inconnu'
    df_infos_joueur['DATE_NAISSANCE'] = pd.to_datetime(df_infos_joueur['DATE_NAISSANCE'])
    df_infos_joueur['DATE_NAISSANCE'] = df_infos_joueur['DATE_NAISSANCE'].dt.strftime('%Y-%m-%d')
    st.write(df_infos_joueur)
    ############################################
    st.subheader("Les statistiques en club")

    df_player["SAISONS_KPI_TOTAL"] = df_player["SAISONS_KPI_TOTAL"].apply(lambda x: x.split(', '))
    df_player["TYPES_KPI_TOTAL"] = df_player["TYPES_KPI_TOTAL"].apply(lambda x: x.split(', '))
    df_player["VALUES_TOTAL"] = df_player["VALUES_TOTAL"].apply(lambda x: x.split(', '))

    # Convert VALUES to numeric values
    df_player["VALUES_TOTAL"] = df_player["VALUES_TOTAL"].apply(lambda x: list(map(int, x)))

    # Use explode to transform lists into separate rows
    df_player_stat = df_player.explode(['SAISONS_KPI_TOTAL', 'TYPES_KPI_TOTAL', 'VALUES_TOTAL'])
    df_player_stat = df_player_stat.rename(columns={'SAISONS_KPI_TOTAL': 'SAISON', 'TYPES_KPI_TOTAL': 'TYPE', "VALUES_TOTAL": 'VALEUR'})

    pivot_df = df_player_stat.pivot_table(index="SAISON", columns="TYPE", values="VALEUR", aggfunc='sum').reset_index()

    # Calculer le nombre de but par minute joué et le nombre de minutes joués par match
    # si la colonne 'NB_BUT' est présente
    if 'NB_BUT' in pivot_df.columns:
        pivot_df['BUTS_PAR_MINUTE'] = pivot_df['NB_BUT'] / pivot_df['NB_MIN_JOUE']
        pivot_df = pivot_df.drop(columns=['NB_BUT'])
    if 'NB_MIN_JOUE' in pivot_df.columns and 'NB_MATCH_JOUE' in pivot_df.columns:
        pivot_df['MINUTES_PAR_MATCH'] = pivot_df['NB_MIN_JOUE'] / pivot_df['NB_MATCH_JOUE']

        pivot_df = pivot_df.drop(columns=['NB_MIN_JOUE'])
        st.write(pivot_df)

    # st.write(df_player.columns)

    ############################################
    # Comparaison avec les autres joueurs
    # prendre les joueurs de bd_app_model qui ont le même poste que le joueur
    # st.write(bd_app_model.columns)
    poste = df_infos_joueur['POSTE'].values[0]
    # si le poste est inconnu, on prend tous les joueurs
    if poste == 'Inconnu':
        bd_comp = bd_app_model
    else:
        bd_comp = bd_app_model[bd_app_model['POSTE'] == poste]

    # age du joueur
    age = 2023 - int(df_infos_joueur['DATE_NAISSANCE'].values[0][:4])
    # st.write(f"Age du joueur : {age}")
    # Si l'age est inférieur ou égale à 16, on choisi les colonnes 'NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16'
    if age <= 16:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_16', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_16']
        bd_comp = bd_comp[col]
    elif age <= 17:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_17', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_17']
        bd_comp = bd_comp[col]
    elif age <= 18:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_18', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_18']
        bd_comp = bd_comp[col]
    elif age <= 19:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_19', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_19']
        bd_comp = bd_comp[col]
    elif age <= 20:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_20', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_20']
        bd_comp = bd_comp[col]
    elif age <= 21:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON_AVANT_21', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON_AVANT_21']
        bd_comp = bd_comp[col]
    else:
        col = ['NB_MIN_JOUE_PAR_MATCH_PAR_SAISON', 'NB_BUT_PAR_MIN_JOUE_PAR_SAISON']
        bd_comp = bd_comp[col]

    df_player_median = df_player[col]
    median_joueur = df_player_median.median()
    # on compare les deux
    st.subheader("Comparaison avec les joueurs sur la période 2013 à 2023")
    st.write("Nous allons comparer les statistiques du joueur avec les autres joueurs ayant le même poste que lui, sur la période 2013 à 2023.")
    st.write("Nous allons comparer les statistiques suivantes :")
    st.write(" - Nombre de minutes jouées par match sur une saison (médiane)")
    st.write(" - Nombre de buts inscrits par minutes jouées sur une saison (médiane)")
    st.write("Les résultats seront affichés sous forme de quartiles, pour chaque statistique. Le joueur sera classé dans un quartile en fonction de ses statistiques.")
    st.write("Le quartile 1 correspond aux joueurs ayant les statistiques les plus faibles, tandis que le quartile 4 correspond aux joueurs ayant les statistiques les plus élevées.")

    def determine_quartile(value, data_column):
        q1 = data_column.quantile(0.25)
        q2 = data_column.quantile(0.50)
        q3 = data_column.quantile(0.75)
        
        if value <= q1:
            return 1
        elif value <= q2:
            return 2
        elif value <= q3:
            return 3
        else:
            return 4

    # res_joueur_quartile = pd.DataFrame()
    # Appliquer la fonction aux variables dans df_player_median

    res_joueur_quartile0 = determine_quartile(median_joueur[col[0]], bd_comp[col[0]])
    res_joueur_quartile1 = determine_quartile(median_joueur[col[1]], bd_comp[col[1]])

    # Créé un dataframe avec les résultats
    res_joueur_quartile = pd.DataFrame({
    'Variable': [
        'Nombre de minutes jouées par match sur une saison (médiane)',
        'Nombre de buts inscrits par minutes jouées sur une saison (médiane)'
    ],
    'VALEUR': [
        median_joueur[col[0]],
        median_joueur[col[1]]
    ],
    'QUARTILE': [
        res_joueur_quartile0,
        res_joueur_quartile1
    ]
})

    # res_joueur_quartile = res_joueur_quartile.rename(columns={col[0]: 'Nombre de minutes jouées par match sur une saison (médiane)',
    #                                                     col[1]: 'Nombre de buts inscrits par minutes jouées sur une saison (médiane)'})
    st.write(res_joueur_quartile)
    #############################################

    st.subheader("Parcours en club")
    df_club = df_player[['SAISONS_EQ_TOTAL', 'EQUIPES_TOTAL']]
    df_club['SAISONS_EQ_TOTAL'] = df_club['SAISONS_EQ_TOTAL'].apply(lambda x: x.split(', '))
    df_club['EQUIPES_TOTAL'] = df_club['EQUIPES_TOTAL'].apply(lambda x: x.split(', '))

    df_club = df_club.explode(['SAISONS_EQ_TOTAL', 'EQUIPES_TOTAL'])
    df_club = df_club.drop_duplicates()
    df_club = df_club.rename(columns={'SAISONS_EQ_TOTAL': 'EQUIPES', 'EQUIPES_TOTAL': 'SAISONS'})
    st.write(df_club)


    
    st.subheader("Parcours en sélection")
    df_selection = df_player[['France U16', 'France U17', 'France U18', 'France U19', 'France U20', 'France U21', 'France']]
    df_selection = df_selection.drop_duplicates()
    # si une des variables vaut autre chose que -1, alor le joueur a été sélectionné
    df_selection_joueur = df_selection[(df_selection != -1).all(1)]
    if df_selection_joueur.empty:
        st.write("Le joueur n'a jamais été sélectionné pour l'instant")
    else:
        st.write(df_selection_joueur)


        

