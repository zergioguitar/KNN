import math
import operator

class KNN( ):
	def __init__(self, train, test, k):
		self.train = train
		self.test = test
		self.k = k
		self.predictions = []

	def predict(self):
		for x in range(len(self.test)):
			aux = neighbors(self.train, self.test[x], self.k)
			result = sumSmiles(aux)
			self.predictions.append(result)

	def accuracy(self):
		achunta = 0
		for x in range(len(self.test)):
			if self.test[x][-1] == self.predictions[x]:
				achunta += 1
		return (achunta/float(len(self.test))) * 100.0

def distance(a, b, length):
	distancia = 0
	for x in range(length):
		distancia += pow((float(a[x]) - float(b[x])), 2)
	return math.sqrt(distancia)

def neighbors(train, testInstance, k):
	length = len(testInstance)-1 #sacamos la columna de tag.
	#calculamos la distancia y la agregamos la agregamos a un arreglo
	distancias = [(train[x], distance(testInstance, train[x], length)) for x in range(len(train))]
	#ordenamos las distancias
	distancias.sort(key=operator.itemgetter(1)) 
	return [distancias[x][0] for x in range(k)]

def sumSmiles(neighbors):
	clases = {} #creamos un diccionario de palabras
	for x in range(len(neighbors)):
		c = neighbors[x][-1] #obtenemos a que clase corresponde.
		if c in clases:
			clases[c] += 1 #aumentamos la cuenta de la clase
		else:
			clases[c] = 1 #agregamos la clase
	#retornamos el valor de las sumas de las clases ordenadas por las clases.
	return sorted(clases.items(), key=operator.itemgetter(1), reverse=True)[0][0]
