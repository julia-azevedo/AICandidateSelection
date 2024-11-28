import streamlit as st
from src.preprocessamento import preprocessar_dados
from src.modelo import treinar_modelo, prever_candidato
import pandas as pd

def main():
    st.title("Sistema de Seleção de Candidatos")
    
    # Entrada de dados do candidato
    nome = st.text_input("Nome do Candidato")
    idade = st.number_input("Idade", min_value=18, max_value=65, step=1)
    nivel_educacao = st.selectbox("Nível de Educação", ['Graduacao', 'Mestrado', 'PhD'])
    anos_experiencia = st.number_input("Anos de Experiência", min_value=0, step=1)
    
    habilidades = st.text_area("Habilidades (separadas por vírgula)", "Python, Machine Learning")
    certificacoes = st.text_area("Certificações (separadas por vírgula)", "AWS")
    analise_comportamental = st.selectbox("Análise Comportamental", ['Proativo', 'Lider', 'Detalhista', 'Comunicativo'])

    if st.button("Analisar"):
        # Carregar dados e preprocessar
        df = pd.read_csv('data/candidatos.csv')
        features, target, le_educacao, le_comportamental = preprocessar_dados(df)
        
        # Treinar modelo
        model = treinar_modelo(features, target)
        
        # Codificar os dados do candidato
        nivel_educacao_cod = le_educacao.transform([nivel_educacao])[0]
        analise_comportamental_cod = le_comportamental.transform([analise_comportamental])[0]
        
        dados_candidato = [idade, anos_experiencia, nivel_educacao_cod, analise_comportamental_cod]
        
        resultado = prever_candidato(model, dados_candidato)
        st.write(f"Resultado: {resultado}")
        st.write("Justificativa:")
        if resultado == "Aprovado":
            st.write("O candidato atende a todos os requisitos do cargo.")
        elif resultado == "Aprovado Parcialmente":
            st.write("O candidato atende à maioria dos requisitos, mas possui algumas lacunas.")
        else:
            st.write("O candidato não atende aos requisitos essenciais.")

if __name__ == "__main__":
    main()
