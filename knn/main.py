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
	ventana.setSize(600,600) 
	ventana.setColor("white")
	ventana.setCloseConfirm()
	ventana.createButton("Browse" , browse)
	return ventana

def browse( ):
	file_name = ventana.askopenfile((("CSV", "*.csv"),("All files", "*.*")))
	test(file_name, True, kMax, sMax)

def readFile(filename, delimiter):
	aux = open(filename).read()
	aux = [item.split(delimiter) for item in aux.split('\n')[:]]
	return aux

def test(database, plot, kNum, sMax):
	ks = []
	accuracies = []

	#leemos la base de datos
	iris = readFile(database, ",") 
	examples = len(iris)
	columns = len(iris[0])

	if(kNum>examples or kNum<1):
		print("Debe usar un k menor a la cantidad de datos y mayor a 0") 
		quit()
	if(sMax<1 or sMax>examples):
		print("Debe usar un s mayor a 0 y menor a la cantidad de datos") 
		quit()
	#desordenamos los datos
	#desordenamos los datos
	indexes = np.random.permutation(examples)
	iris = np.asarray(iris)[indexes].tolist()

	elemSeg =  len(iris)/sMax
	segmentos = [iris[int(i*elemSeg):int(i*elemSeg+elemSeg)] for i in range(sMax)]
	
	#probamos para k desde 1 a 10.
	for k in range(1,kNum+1): #el ultimo termino es exclusivo
	    accuracy = 0
	    for i in range(sMax): 
	    	#seleccionamos los grupos de entrenamiento y los de prueba.
	    	#se hará 10 veces, para generar una evaluacion cruzada
	    	index = np.delete([x for x in range(sMax)], i)
	    	train = np.concatenate(np.asarray(segmentos)[index].tolist())
	    	test = segmentos[i]

	    	knn = KNN(train,test,k) #train data, test data, k value
	    	knn.predict()
	    	accuracy += knn.accuracy()
	    accuracy /= sMax
	    print("Accuracy: " , accuracy, "k: " , k)
	    if(plot):
	    	ks.append(k);
	    	accuracies.append(accuracy)
	    	
	if(plot):
		ventana.deleteAllButton()
		ventana.graphPlot(ks,accuracies,"k's","Accuracy","Accuracy Graph")


#######  Main Program  #######
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', metavar='filename', action="store", dest="file", help='database file path name.')
    parser.add_argument('-k', metavar='k', action="store", dest="k", help='max k number to test (Default = 10)')
    parser.add_argument('-s', metavar='splits', action="store", dest="s", help='number of segments to validate (Default = 10)')
    args = parser.parse_args()
    kMax = 10
    if(args.k != None):
    	kMax= int(args.k)

    sMax = 10
    if(args.s != None):
    	sMax= int(args.s)

    if(args.file == None):
    	ventana = GUI()
    	ventana.keepAlive()
    else:
    	test(args.file, False, kMax, sMax)
