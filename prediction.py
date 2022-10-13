from srcs.perceptron import  Perceptron
import pandas as pd

def main():
    try :
        df = pd.read_csv("./data.csv")
    except:
        raise Exception("Error while opening data.csv")

    X = df.iloc[0:,0].values
    y = df.iloc[0:,1].values

    mean = X[0:].mean() # Moyenne des données
    std = X[0:].std() # Écart type
    X_std = (X[0:] - mean) / std # Normalize data
    percep = Perceptron()

    try:
        df = pd.read_csv('./model.csv', header=None)
        percep._theta0 = df.iloc[0, 0]
        percep._theta1 = df.iloc[0, 1]
    except:
        print("Model.csv not found or corrupted.")
        pass

    km_search = input('Kilometrage ? : ')
    while (km_search.isnumeric() == 0):
        print("Input invalide")
        km_search = input('Kilometrage ? : ')

    km_search = (float(km_search) - mean) / std
    predicted_price = percep.predict(km_search)
    if predicted_price < 0:
        predicted_price = 0.0
    print(predicted_price)

if __name__ == "__main__":
    main()