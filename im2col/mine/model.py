from layers import *


class CNN:
    def __init__(self, input_dim=(1, 28, 28), hidden_size=50, output_size=10):
        pre_node_nums = np.array(
            [1 * 3 * 3,
             16 * 3 * 3,
             16 * 3 * 3,
             hidden_size # 凯明初始化用
             ]
        )
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)

        conv_param1 = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param2 = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        conv_param3 = {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}

        self.params = {}

        self.params['W1'] = weight_init_scales[0] * np.random.randn(conv_param1['filter_num'], input_dim[0],
                                                                    conv_param1['filter_size'],
                                                                    conv_param1['filter_size'])
        self.params['b1'] = np.zeros(conv_param1['filter_num'])

        self.params['W2'] = weight_init_scales[1] * np.random.randn(conv_param2['filter_num'],
                                                                    conv_param1['filter_num'],
                                                                    conv_param2['filter_size'],
                                                                    conv_param2['filter_size'])
        self.params['b2'] = np.zeros(conv_param2['filter_num'])

        self.params['W3'] = weight_init_scales[2] * np.random.randn(conv_param3['filter_num'],
                                                                    conv_param2['filter_num'],
                                                                    conv_param3['filter_size'],
                                                                    conv_param3['filter_size'])
        self.params['b3'] = np.zeros(conv_param3['filter_num'])

        self.params['W4'] = weight_init_scales[3] * np.random.randn(hidden_size, output_size)
        self.params['b4'] = np.zeros(output_size)