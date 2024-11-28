import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessar_dados(df):
    le_educacao = LabelEncoder()
    df['Nivel_Educacao_Cod'] = le_educacao.fit_transform(df['Nivel_Educacao'])

    le_comportamental = LabelEncoder()
    df['Analise_Comportamental_Cod'] = le_comportamental.fit_transform(df['Analise_Comportamental'])

    features = df[['Idade', 'Anos_Experiencia', 'Nivel_Educacao_Cod', 'Analise_Comportamental_Cod']]
    target = df['Resultado']

    return features, target, le_educacao, le_comportamental
