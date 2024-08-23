import streamlit as st
import pandas as pd
import joblib

def main():
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
    unsafe_allow_html=True)

    st.title("Prédiction du parcours des joueurs en sélection nationale")
    st.markdown("Auteur : Clémence CHESNAIS")
    st.write(
        "Cette application a été élaborée dans le but de prédire si un joueur a des chances de jouer dans l'une des équipes de France.")
    st.write(
        "Pour ce faire, plusieurs modèles de machine learning ont été entraînés sur un jeu de données contenant les informations de joueurs ayant joué dans les équipes de France de jeunes (U16, U17, U18, U19, U20, U21) et dans l'équipe de France A, durant la période de 2013 à 2023.")
    st.write(
        "L'utilisateur importe un fichier CSV contenant les informations des joueurs."
        " L'utilisateur sélectionne le nom du joueur ainsi que la catégorie U des sélections dans laquelle il est actuellement ('Aucune' s'il ne fait partie d'aucune des équipes de France)."
    )
    st.write(
        "En appuyant sur le bouton 'PREDIRE', la prédiction s'affiche :") 
    st.write(    
        " - Classe **1** : le joueur a des chances de jouer dans la catégorie d'âge sélectionnée.")
    st.write(    
        " - Classe **0** : le joueur n'a (pour l'instant) pas de chance (d'après le modèle de prédiction).") 
    st.write(
        "On affichera également la probabilité que le joueur soit sélectionné dans cette catégorie d'âge.")

    
    

    

if __name__ == '__main__':
    main()