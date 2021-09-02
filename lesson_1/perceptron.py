import numpy as np


class Perceptron:
    def __init__(self, data, class_numb, neuron_numb=[5]):
          self.neuron_numb = neuron_numb
          self.weights = [np.random.uniform(-1, 1, (data.shape[1], neuron_numb[0]))]
          self.weights.extend([np.random.uniform(-1, 1, (neuron_numb[i], neuron_numb[i+1])) for i in range(len(neuron_numb)-1)])
          self.weights.append(np.random.uniform(-1, 1, (neuron_numb[-1], class_numb)))

          self.dict_func = {'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
                            'tanh': lambda x: (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)),
                            'relu': lambda x: x*(x > 0)}
          self.dict_grad = {'sigmoid': lambda x: self.func(x, name='sigmoid') * (1 - self.func(x, name='sigmoid')),
                            'tanh': lambda x: 1 - self.func(x, name='tanh') ** 2,
                            'relu': lambda x: x >= 0}

    def func(self, x, name='sigmoid'):
          return self.dict_func[name](x)

    def grad(self, x, name='sigmoid'):
          return self.dict_grad[name](x)

    def fit(self, data, y_train, eras=1000, learn_r=0.05, act_func='sigmoid'):
          errors = []
          accuracy = 0

          for i in range(eras):
                # прямое распространение(feed forward)
                layers = [data]
                for l in range(1, len(self.neuron_numb)+2):
                      layers.append(self.func(np.dot(layers[l-1], self.weights[l-1]), name=act_func))
                layers_rev = layers.copy()
                layers_rev.reverse()

                # обратное распространение(back propagation) с использованием градиентного спуска
                w_rev = self.weights.copy()
                w_rev.reverse()
                layer_error = (y_train - layers_rev[0])
                layers_delta = [(y_train - layers_rev[0]) * self.grad(layers_rev[0])]
                for i in range(len(self.weights)-1):
                      layers_delta.append(layers_delta[i].dot(w_rev[i].T)*self.grad(layers_rev[i+1]))

                # коррекция
                for ind,w in enumerate(w_rev):
                      w += layers_rev[ind+1].T.dot(layers_delta[ind])*learn_r
                self.weights = w_rev.copy()
                self.weights.reverse()


                # метрика модели
                error = np.mean(np.abs(layer_error))
                errors.append(error)
                accuracy = (1 - error) * 100

          print(f"Аккуратность нейронной сети {round(accuracy, 2)}%")