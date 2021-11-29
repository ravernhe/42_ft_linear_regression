from srcs.perceptron import  Perceptron
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def show_model(percep, X_std, X, y):
  plt.scatter(X, y, marker='x')
  plt.plot(X, percep._theta1 * X_std + percep._theta0)
  plt.title('Linear Regression')
  plt.xlabel('Kilométrage')
  plt.ylabel('Prix')
  plt.show()

def main(visual=False,precision=False):
    df = pd.read_csv("./data.csv")

    X = df.iloc[0:,0].values
    y = df.iloc[0:,1].values

    mean = X[0:].mean() # Moyenne des données
    std = X[0:].std() # Écart type
    X_std = (X[0:] - mean) / std # Normalize data

    percep = Perceptron(eta=0.15,n_iter=19) 
    percep.fit(X_std, y)

    percep.save_model()

    # Si R est proche de 1 : il y a une forte liaison linéaire entre les variables et les valeurs prises par Y ont tendance à croître quand les valeurs de X augmentent.
    # Si R est proche de 0 : il n’y a pas de liaison linéaire
    # Si R est proche de -1 : il y a une forte liaison linéaire et les valeurs prises par Y ont tendance à décroître quand les valeurs de X augmentent.
    
    if (precision):
        print("Model precison : ",percep.get_precision(X_std, y))
    if (visual):
        show_model(percep, X_std, X, y)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', "--visual", help="Show model")
    parser.add_argument('-p', "--precision", help="Algo precision")
    args = parser.parse_args()
    main(args.visual, args.precision)