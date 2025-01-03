import numpy as np
import layer


class CNN():
    
    def __init__(self, input_dim=(1, 28, 28),hidden_size=256, output_size=10):
        
        pre_node_nums = np.array([1*3*3, 16*3*3, 32*3*3, 64*3*3,64*7*7,hidden_size,128,64])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用ReLU的情况下推荐的初始值
        
        conv_param1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param2 = {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1}
        conv_param3 = {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1}
        conv_param4 = {'filter_num':64, 'filter_size':4, 'pad':1, 'stride':1}
        self.params = {}
        
        self.params['W1'] = weight_init_scales[0] * np.random.randn(conv_param1['filter_num'],input_dim[0],conv_param1['filter_size'],conv_param1['filter_size'])
        self.params['b1'] = np.zeros(conv_param1['filter_num'])
        
        self.params['W2'] = weight_init_scales[1] * np.random.randn(conv_param2['filter_num'],conv_param1['filter_num'],conv_param2['filter_size'],conv_param2['filter_size'])
        self.params['b2'] = np.zeros(conv_param2['filter_num'])
        
        self.params['W3'] = weight_init_scales[2] * np.random.randn(conv_param3['filter_num'],conv_param2['filter_num'],conv_param3['filter_size'],conv_param3['filter_size'])
        self.params['b3'] = np.zeros(conv_param3['filter_num'])
        
        self.params['W4'] = weight_init_scales[3] * np.random.randn(conv_param4['filter_num'],conv_param3['filter_num'],conv_param4['filter_size'],conv_param4['filter_size'])
        self.params['b4'] = np.zeros(conv_param4['filter_num'])
        
        self.params['W5'] = weight_init_scales[4] * np.random.randn(64*7*7, hidden_size)
        self.params['b5'] = np.zeros(hidden_size)
        
        self.params['W6'] = weight_init_scales[5] * np.random.randn(hidden_size,128)
        self.params['b6'] = np.zeros(128)
        
        self.params['W7'] = weight_init_scales[6] * np.random.randn(128,64)
        self.params['b7'] = np.zeros(64)
        
        self.params['W8'] = weight_init_scales[7] * np.random.randn(64,output_size)
        self.params['b8'] = np.zeros(output_size)
    
        
        
        
        
        self.layers = []
        self.layers.append(layer.Convolution(self.params['W1'], self.params['b1'], conv_param1['stride'], conv_param1 ['pad'])) # 0
        self.layers.append(layer.Relu())
        self.layers.append(layer.Convolution(self.params['W2'], self.params['b2'], conv_param2['stride'], conv_param2['pad'])) # 2
        self.layers.append(layer.Relu())
        self.layers.append(layer.MaxPooling(pool_h=2, pool_w=2, stride=2))
        
        self.layers.append(layer.Convolution(self.params['W3'], self.params['b3'], conv_param3['stride'], conv_param3['pad'])) # 5
        self.layers.append(layer.Relu())
        self.layers.append(layer.Convolution(self.params['W4'], self.params['b4'], conv_param4['stride'], conv_param4['pad'])) #7
        self.layers.append(layer.Relu())
        self.layers.append(layer.MaxPooling(pool_h=2, pool_w=2, stride=2))
        
        
        
        self.layers.append(layer.Affine(self.params['W5'],self.params['b5'])) #10
        self.layers.append(layer.Relu())
        self.layers.append(layer.Affine(self.params['W6'],self.params['b6'])) #12
        self.layers.append(layer.Relu())
        self.layers.append(layer.Affine(self.params['W7'],self.params['b7'])) #14
        self.layers.append(layer.Relu())
        self.layers.append(layer.Affine(self.params['W8'],self.params['b8'])) #16
        
        self.layers.append(layer.Softmax())
        
    def forward(self,x):

        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    #def loss(self,x,t):
        
    #    return self.layers[-1].soft_loss(x,t)
    
    def gradient(self,t):
        
        dout = 1

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        dout = tmp_layers[0].backward(t,dout)
        for layer in tmp_layers[1:]:
            dout = layer.backward(dout)
            
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12,14,16)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db 
            
        return grads 
    
    def accuracy(self, y, t):
        acc = 0.0
        length =y.shape[0]

        y_max = np.argmax(y,axis=1)
        t_max = np.argmax(t,axis=1)

        acc = np.sum(y_max==t_max)/length

        return acc

    def loadparams(self,params):
        i = 0
        keys = list(params.keys())
        for temp in self.layers:
            if isinstance(temp, layer.Convolution) or isinstance(temp, layer.Affine):
                temp.W = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1
                temp.b = params[keys[i]]
                self.params[keys[i]] = params[keys[i]]
                i+=1


    
    
            
                
        
        
        