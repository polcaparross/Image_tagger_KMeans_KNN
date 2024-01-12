__authors__ = ['1633753', '1633822', '1633937']
__group__ = 'DL.10'


import numpy as np
import utils

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X): #DONE
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        
        X = np.array(X)
        
        if X.dtype != np.float: #convertim a float si no ho és
            X = X.astype(float)
        
        #convertim en array de 2 dimensions
        if X.ndim == 1: # 
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])
            
        self.X = X
        
    

    def _init_options(self, options=None): #DONE
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
    
    def _init_centroids(self): #DONE
        """
        Initialization of centroids
        """
        self.centroids = np.zeros((self.K, self.X.shape[-1])) #crea matrix tamaño K x D amb zeros
      
        if self.options['km_init'] == 'first':
            unics = []
            for row in self.X:
                if not any((row == x).all() for x in unics): #si trobem un punt que encara no està a unics l'afegim.
                    unics.append(row)
                if len(unics) == self.K: #si ja hem seleccionat els primer k punts ja hem acabat
                    break

            if len(unics) < self.K:
                print("Els punts únics son menors que K")
            else:
                self.centroids = np.array(unics) #assignem els primers k punts d'X
        
        elif self.options['km_init'] == 'random':
            indexs = np.random.choice(self.X.shape[0], self.K, replace=False) #generem k indexs de forma aleatoria
            self.centroids = self.X[indexs] 
            
            for i in range(len(self.K)):
                self.centroids = np.random.rand(self.K, self.X.shape[1])
                self.old_centroids = np.random.rand(self.K, self.X.shape[1])
    
        elif self.options['km_init'] == 'custom':
            #incialitzem els centroides agafant els primers k punts unics pero començant pel final
            unics = []
            for i in range(len(self.X)-1, -1, -1):
                if not any((self.X[i] == x).all() for x in unics): #si trobem un punt que encara no està a unics l'afegim.
                    unics.insert(0, self.X[i])
                if len(unics) == self.K: #si ja hem seleccionat els primer k punts (començant pel final) ja hem acabat
                    break
                
            if len(unics) < self.K:
                print("Els punts únics son menors que K")
            else:
                self.centroids = np.array(unics) #assignem els punts

            
        


    def get_labels(self):  #DONE
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """       
        self.labels = np.argmin(distance(self.X, self.centroids),axis=1)
        #calculem la distancia de cada punt d'X amb cada un dels centroides i a labels guardem a cada index l'index del centroide més proper a aquell punt
        #argmin retorna l'index del valor minim
    
    def get_centroids(self): #DONE
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids #guardem els centroides anteriors abans de calcular els nous
        self.get_labels()   
        llista = np.empty([self.K, 3])

        for k in range(self.K): #bucle x calcular els centroides de cada k 
            v = np.array(self.X[np.where(self.labels == k)]) #v es un array amb els indices dels punts de self.X que pertanyen al cluster k
            total = v.shape[0]
            suma_x = np.sum(v[:, 0])
            suma_y = np.sum(v[:, 1])
            suma_z = np.sum(v[:, 2])
            #calculem la mitja de les 3 dimensions i els guardem com a nous centroides
            llista[k] = suma_x/total, suma_y/total, suma_z/total

        self.centroids = llista
    
    

    def converges(self): #DONE
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, atol = self.options['tolerance'])
        #calculem si els centroids i old_centroids son iguals utilitzant la funcio allclose de numpy

    
    def fit(self): #DONE
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        i = 0
        self._init_centroids() #inicialitzem els centroides

        while True:
            self.get_centroids() #per cada punt troba el centroide més proper
            if self.options['max_iter'] == i or self.converges(): 
                break #acabem si hem fet les màximes iteracions o centroid i old_centroids coincideixen
            i+=1 #si no hem acabat augmentem el nombre d'iteracions
        	
            
            
        
    def withinClassDistance(self): #DONE
        """
         returns the within class distance of the current clustering
         (calcul dist intra-class)
        """
    
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        suma = 0

        for x in range(len(self.centroids)):
            suma += np.sum((np.square( self.X[np.where(self.labels == x)] - self.centroids[x]) )**2 ) #formula distancia intra class

       	return suma*self.K
    
    def find_bestK(self, max_K): #DONE
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        wcd_llista = [] #llista on anirem guardant la distancia intra class
        wcd_previ = None #variable per guardar la distancia intra class anterior que utilitzarem per calcular el % de decreixement
        best_K = max_K #en cas que no trobem k ideal, aquesta serà la k màxima

        for k in range(2, max_K + 1): #recorrem tots els clusters (el minim és 2)
            kmeans = KMeans(X=self.X, K=k) #creem objecte classe kmeans
            kmeans.fit() #apliquem l'algorisme
            wcd = kmeans.withinClassDistance() #calculem la dist intra class i la guardem a la llista
            wcd_llista.append(wcd) 
            if wcd_previ is not None: 
                dec = 100 * (wcd_previ - wcd) / wcd_previ #calculem el decreixement
                if dec < 20: #llindar recomenat a l'enunciat (20%)
                    best_K = k - 1 #si trobem una k que passi de decrement alt a estabilitzat (per sota del llindar (20%)) tenim k ideal
                    break
            wcd_previ = wcd #actualitzem el wcd previ per la seguent k

        self.K = best_K
        return best_K


def distance(X, C): #DONE
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    
    return np.linalg.norm(np.expand_dims(X, 2) - np.expand_dims(C.T, 0), axis=1)
    #con expand_dims añadimos a X una nueva dimension al final y a C.T una delante para poder hacer la resta X = (PxDx1) y C.T (1xKxD)
    #linalg.norm hace la norma euclidiana para calcular la distancia
    

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors (li hem dit c_centroides per no confondrens)
    """

    prob_c = utils.get_color_prob(centroids) #calculem la pertinença
    c = np.array(['Red','Orange','Brown','Yellow','Green','Blue','Purple','Pink','Black','Grey','White']) #inicialitzem array numpy amb els 11 colors
    c_centroides = []
    
    for i in range(len(centroids)):
        
    	p = 0
    	i_color = 0
        
    	for j in range(len(prob_c[i])):
    		if p < prob_c[i][j]:
    			p = prob_c[i][j] #calculem per cada centroide el color amb el que te major pertinença (probabilitat)
    			i_color = j
    	c_centroides.append(c[i_color]) #afegim a la llista el color corresponent del centroide

    return c_centroides