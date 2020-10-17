import numpy as np
class Dann:
    def __init__(self,i,o):
        self.i = i
        self.inputs = Matrix(i,1)

        self.o = o
        self.outputs = Matrix(o,1)

        self.aFunc = []
        self.aFunc_d = []

        self.Layers = [self.inputs,self.outputs]
        self.weights = []
        self.biases = []
        self.errors = []
        self.gradients = []

        self.outs = []
        self.loss = 0
        self.losses = []
        self.lr = 0.001
        self.arch = [];
    def log(self):
        l = len(self.Layers)
        for i in range(l):
            if i == 0:
                print("inputs  : "+str(self.Layers[i].rows))
            elif i == l-1:
                print("outputs : "+str(self.Layers[i].rows)+"  " + str(self.aFunc[i-1]))
            else:
                print("layer   : "+str(self.Layers[i].rows)+"  " + str(self.aFunc[i-1]))
        for i in range(len(self.weights)):
            self.weights[i].log()
        print(self.lr)


    def addHiddenLayer(self,size,act):
        layer = Matrix(size,1)
        index = len(self.Layers)-2
        l = len(self.Layers)-1

        self.Layers.insert(l,layer)

        if act != None:
            nor = act.__name__
            der = act.__name__ + "_d"
            self.aFunc.insert(index,nor)
            self.aFunc_d.insert(index,der)
    def makeWeights(self):
        for i in range(len(self.Layers)-1):
            weights = Matrix(self.Layers[i+1].rows,self.Layers[i].rows)
            biases = Matrix(self.Layers[i+1].rows,1)

            weights.initiate()
            biases.initiate()

            self.weights.insert(i,weights)
            self.biases.insert(i,biases)

            self.errors.insert(i, Matrix(self.Layers[i+1].rows,1))
            self.gradients.insert(i,Matrix(self.Layers[i+1].rows,1))

            if len(self.aFunc)-1 < i:
                self.aFunc.insert(i,"sigmoid")
                self.aFunc_d.insert(i,"sigmoid_d")
    def calcMeanLoss(self,arr,t):
        sum = 0
        ans = 0
        for i in range(self.o):
            sum += np.power(arr[i] - t[i],2)
        ans = sum/self.o
        return ans
    def feedForward(self,inputs):
        self.Layers[0] = Matrix.fromArray(inputs)

        for i in range(len(self.weights)):
            self.Layers[i+1] = Matrix.multiply(self.weights[i],self.Layers[i])
            self.Layers[i+1].add(self.biases[i])
            self.Layers[i+1].map(self.aFunc[i])
        self.outs = []
        for i in range(self.o):
            self.outs.insert(i,Matrix.toArray(self.Layers[len(self.Layers)-1])[i])
        return self.outs
    def backpropagate(self,inputs,t):
        targets = Matrix.fromArray(t)

        self.feedForward(inputs);
        llength = len(self.Layers)-1
        self.errors[len(self.errors)-1] = Matrix.subtract(targets, self.Layers[llength])
        glength = len(self.gradients)-1
        funclength = len(self.aFunc)-1
        elength = len(self.errors)-1
        self.gradients[glength] = Matrix.mapS(self.Layers[llength],self.aFunc_d[funclength])
        self.gradients[glength].mult(self.errors[elength])
        self.gradients[glength].mult(self.lr)

        i = len(self.weights)-1
        for x in range(i):

            h_t = Matrix.transpose(self.Layers[i])
            weights_deltas = Matrix.multiply(self.gradients[i],h_t)

            self.weights[i].add(weights_deltas)
            self.biases[i].add(self.gradients[i])

            weights_t = Matrix.transpose(self.weights[i])
            self.errors[i-1] = Matrix.multiply(weights_t,self.errors[i])

            self.gradients[i-1] = Matrix.mapS(self.Layers[i], self.aFunc_d[i-1])

            self.gradients[i-1].mult(self.errors[i-1])
            self.gradients[i-1].mult(self.lr)

            i-=1

        i_t = Matrix.transpose(self.Layers[0])
        weights_deltas = Matrix.multiply(self.gradients[0], i_t)

        self.weights[0].add(weights_deltas)
        self.biases[0].add(self.gradients[0])

        self.loss = self.calcMeanLoss(self.outs,t)
        self.losses.append(self.loss)


class Matrix:
    def __init__(self,rows,cols):
        self.rows = rows
        self.cols = cols
        self.matrix = []
        for i in range(rows):
            c = [];
            for j in range(cols):

                c.append(0)

            self.matrix.append(c)
    def log(self):
        for i in range(self.rows):
            print(self.matrix[i])

        print('')
    def add(self,n):
        if isinstance(n,Matrix):
            if self.rows != n.rows or self.cols != n.cols:
                print('rows of A must be equal to rows of B')
                return 0
            else:
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.matrix[i][j] += n.matrix[i][j]
        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] += n
    @staticmethod
    def transpose(m):
        ans = Matrix(m.cols,m.rows)

        for i in range(m.rows):
            for j in range(m.cols):
                ans.matrix[j][i] = m.matrix[i][j]
        return ans
    @staticmethod
    def mapS(m,f):
        for i in range(m.rows):
            for j in range(m.cols):
                v = m.matrix[i][j]
                possibles = globals().copy()
                possibles.update(locals())
                method = possibles.get(f)
                m.matrix[i][j] = method(v)
        return m
    def map(self,f):
        for i in range(self.rows):
            for j in range(self.cols):
                v = self.matrix[i][j]
                possibles = globals().copy()
                possibles.update(locals())
                method = possibles.get(f)
                self.matrix[i][j] = method(v)

    @staticmethod
    def fromArray(arr):
        l = len(arr)
        m = Matrix(l,1)
        for i in range(l):
            m.matrix[i][0] = arr[i]
        return m
    @staticmethod
    def multiply(a,b):
        if a.cols != b.rows:
            print('cols of A must be equal to rows of B...(static)')
            return 0
        else:
            ans = Matrix(a.rows,b.cols)
            for i in range(ans.rows):
                for j in range(ans.cols):
                    sum = 0
                    k = 0
                    for k in range(a.cols):
                        sum += a.matrix[i][k] * b.matrix[k][j]
                        ans.matrix[i][j] = sum
            return ans
    @staticmethod
    def subtract(a,b):
        ans = Matrix(a.rows,a.cols)
        for i in range(ans.rows):
            for j in range(ans.cols):
                ans.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j]

        return ans
    @staticmethod
    def toArray(m):
        ans = [];
        if m.cols == 1:
            for i in range(m.rows):
                ans.insert(i,m.matrix[i][0])
        return ans
    def mult(self,n):

        if isinstance(n,Matrix):
            if self.cols != n.cols or self.rows != n.rows:
                print('cols of A must be equal to rows of B...(self):'+str(self.matrix)+" : "+str(n.matrix))
                return None
            else:
                ans = Matrix(self.rows,n.cols)
                for i in range(ans.rows):
                    for j in range(ans.cols):
                        sum = 0
                        k = 0
                        for k in range(self.cols):
                            self.matrix[i][k] *= n.matrix[k][j]


        else:
            for i in range(self.rows):
                for j in range(self.cols):
                    self.matrix[i][j] *= n

    def set(matrix):
        self.matrix = matrix
    def initiate(self):
        for i in range(self.rows):
            for j in range(self.cols):
                w = np.random.random_sample();
                w = np.round(w*10)/10;
                self.matrix[i][j] = w;

def sigmoid(x):
    return 1/(1+(np.exp(-x)))
def sigmoid_d(x):
    return sigmoid(x)*(1 - sigmoid(x))
def leakyRelu(x):
    if x > 0:
        return x
    else:
        return x*0.01
def leakyRelu_d(x):
    if x > 0:
        return 1
    else:
        return 0.01
