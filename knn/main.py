############################################
###            Desarrollado por          ###
###       Sergio Elías Flores Labra      ###
###         serflo.elec@gmail.com        ###
############################################
###        Optimizado para Python 3		 ###
############################################
###          Requisitos:				 ###
###         	numpy					 ###
###         	tkinter					 ###
############################################


#######  Imports  #######
from sfloresLib import *
from knn import *
import numpy as np
import argparse 

#######  Functions  #######
def GUI( ):
	ventana = createWindow()
	ventana.setTitle("KNN")
	ventana.setSize(400,400) 
	ventana.setColor("white")
	ventana.setCloseConfirm()
	ventana.createButton("Browse" , browse)
	return ventana

def browse( ):
	file_name = ventana.askopenfile((("CSV", "*.csv"),("All files", "*.*")))
	test(file_name, True)

def readFile(filename, delimiter):
	aux = open(filename).read()
	aux = [item.split(delimiter) for item in aux.split('\n')[:-1]]
	return aux

def test(database, plot):
	#leemos la base de datos
	iris = readFile(database, ",") 
	examples = len(iris)
	columns = len(iris[0])

	#desordenamos los datos
	indexes = np.random.permutation(examples)
	iris = np.asarray(iris)[indexes].tolist()

	#dividimos los datos en 90% y 10%
	train = iris[:int(examples*0.9)]
	test = iris[int(examples*0.9):examples]

	for k in range(1,11): #el ultimo termino es exclusivo
		knn = KNN(train,test,k) #train data, test data, k value
		knn.predict()
		if(plot):
			ventana.createLabel( "Accurracy: " + str(knn.accuracy()) + " k: " + str(k) )
		else:
			print("Accuracy: " , knn.accuracy(), "k: " , k)

#######  Main Program  #######
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='filename', action="store", dest="file", help='database file path name.')
    args = parser.parse_args()
    if(args.file == None):
    	ventana = GUI()
    	ventana.keepAlive()
    else:
    	test(args.file, False)
