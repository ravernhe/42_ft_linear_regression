class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta #Learning rate (between 0.0 and 1.0)
        self.n_iter = n_iter #Passes over the training dataset
        self._theta0 = 0.0
        self._theta1 = 0.0

    def fit(self, X, y):
        self._theta0 = 0.0
        self._theta1 = 0.0

        for _ in range (self.n_iter):
            tmp0 = 0
            tmp1 = 0

            for kilometrage,prix in zip(X, y):
                risk = self.predict(kilometrage) - prix
                tmp0 += risk 
                tmp1 += risk * kilometrage
            self._theta0 -= self.eta / len(X) * tmp0
            self._theta1 -= self.eta / len(X) * tmp1
        return self

    def predict(self, X): # prixEstime(Kilométrage) = theta0 + (theta1 * km)
        return self._theta0 + (self._theta1 * X)

    def save_model(self, filename='./model.csv'):
        f = open(filename, 'w+')
        f.write(f'{self._theta0},{self._theta1}')
        f.close()
        return self

    def get_precision(self, X, y):
        top = 0.0
        bottom = 0.0
        sum_y = 0.0

        for prix in y :
            sum_y += prix
        sum_y /= len(y)

        for kilometrage,prix in zip(X, y):
                expected = self.predict(kilometrage)
                top += pow((prix - expected), 2)
                bottom += pow((prix - sum_y), 2)
        
        rse = top / bottom

        return (1 - rse)