from sklearn.ensemble import RandomForestClassifier

def treinar_modelo(features, target):
    model = RandomForestClassifier()
    model.fit(features, target)
    return model

def prever_candidato(model, dados):
    predicao = model.predict([dados])
    return predicao[0]
