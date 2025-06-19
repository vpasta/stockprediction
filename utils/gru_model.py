import numpy as np
from .activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative 

class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.is_training = True 

        self.Wz = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bz = np.zeros((1, hidden_dim))

        self.Wr = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.br = np.zeros((1, hidden_dim))

        self.Wh = np.random.randn(input_dim, hidden_dim) * 0.01
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh = np.zeros((1, hidden_dim))

        self.Wo = np.random.randn(hidden_dim, output_dim) * 0.01
        self.bo = np.zeros((1, output_dim))

        self.cache = {}

    def forward(self, X):
        """
        Melakukan forward pass pada GRU untuk satu sequence.

        Args:
            X (np.ndarray): Input sequence, shape (sequence_length, input_dim)
        """
        sequence_length = X.shape[0]
        
        h_prev = np.zeros((1, self.hidden_dim))

        hidden_states = []
        sequence_cache = []
        
        for t in range(sequence_length):
            x_t = X[t].reshape(1, self.input_dim)

            z_pre = np.dot(x_t, self.Wz) + np.dot(h_prev, self.Uz) + self.bz
            r_pre = np.dot(x_t, self.Wr) + np.dot(h_prev, self.Ur) + self.br
            
            z_t = sigmoid(z_pre)
            r_t = sigmoid(r_pre)
            
            h_tilde_pre = np.dot(x_t, self.Wh) + np.dot(r_t * h_prev, self.Uh) + self.bh
            h_tilde_t = tanh(h_tilde_pre)

            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t
            
            dropout_mask = np.ones_like(h_t)
            if self.is_training and self.dropout_rate > 0:
                # Inverted dropout
                dropout_mask = (np.random.rand(*h_t.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                h_t *= dropout_mask

            sequence_cache.append({
                'x_t': x_t,
                'h_prev': h_prev.copy(), 
                'z_pre': z_pre, 'r_pre': r_pre, 'h_tilde_pre': h_tilde_pre,
                'z_t': z_t,
                'r_t': r_t,
                'h_tilde_t': h_tilde_t,
                'h_t': h_t,
                'dropout_mask': dropout_mask
            })

            h_prev = h_t
            hidden_states.append(h_t)

        final_h_t = hidden_states[-1]
        output = np.dot(final_h_t, self.Wo) + self.bo

        self.cache['sequence_cache'] = sequence_cache
        self.cache['final_h_t'] = final_h_t
        self.cache['output'] = output
        return output

    def backward(self, d_output, learning_rate):
        sequence_cache = self.cache['sequence_cache']
        final_h_t = self.cache['final_h_t']

        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)

        dWo += np.dot(final_h_t.T, d_output)
        dbo += d_output

        dh_next = np.dot(d_output, self.Wo.T)

        for t in reversed(range(len(sequence_cache))):
            cache_t = sequence_cache[t]
            x_t, h_prev, z_pre, r_pre, h_tilde_pre, z_t, r_t, h_tilde_t, h_t, dropout_mask = \
                cache_t['x_t'], cache_t['h_prev'], cache_t['z_pre'], cache_t['r_pre'], cache_t['h_tilde_pre'], \
                cache_t['z_t'], cache_t['r_t'], cache_t['h_tilde_t'], cache_t['h_t'], cache_t['dropout_mask']

            dh_t = dh_next * dropout_mask 

            dh_prev_term1 = dh_t * (1 - z_t)

            dz_t = dh_t * (h_tilde_t - h_prev)
            dz_pre = dz_t * sigmoid_derivative(z_pre)

            dh_tilde_t = dh_t * z_t
            dh_tilde_pre = dh_tilde_t * tanh_derivative(h_tilde_pre)

            dWh += np.dot(x_t.T, dh_tilde_pre)
            dbh += dh_tilde_pre
            
            dUh += np.dot((r_t * h_prev).T, dh_tilde_pre)
            
            dr_t_from_h_tilde = np.dot(dh_tilde_pre, self.Uh.T) * h_prev

            dr_t = dr_t_from_h_tilde
            dr_pre = dr_t * sigmoid_derivative(r_pre)

            dWr += np.dot(x_t.T, dr_pre)
            dbr += dr_pre
            dUr += np.dot(h_prev.T, dr_pre)

            dWz += np.dot(x_t.T, dz_pre)
            dbz += dz_pre
            dUz += np.dot(h_prev.T, dz_pre)

            dh_next = dh_prev_term1 + \
                      np.dot(dz_pre, self.Uz.T) + \
                      np.dot(dr_pre, self.Ur.T) + \
                      (np.dot(dh_tilde_pre, self.Uh.T) * r_t)


        self.grads = {
            'dWz': dWz, 'dUz': dUz, 'dbz': dbz,
            'dWr': dWr, 'dUr': dUr, 'dbr': dbr,
            'dWh': dWh, 'dUh': dUh, 'dbh': dbh,
            'dWo': dWo, 'dbo': dbo
        }

    def update_weights(self, learning_rate):
        self.Wz -= learning_rate * self.grads['dWz']
        self.Uz -= learning_rate * self.grads['dUz']
        self.bz -= learning_rate * self.grads['dbz']

        self.Wr -= learning_rate * self.grads['dWr']
        self.Ur -= learning_rate * self.grads['dUr']
        self.br -= learning_rate * self.grads['dbr'] 

        self.Wh -= learning_rate * self.grads['dWh']
        self.Uh -= learning_rate * self.grads['dUh']
        self.bh -= learning_rate * self.grads['dbh']

        self.Wo -= learning_rate * self.grads['dWo']
        self.bo -= learning_rate * self.grads['dbo']

    def get_weights(self):
        return {
            'Wz': self.Wz, 'Uz': self.Uz, 'bz': self.bz,
            'Wr': self.Wr, 'Ur': self.Ur, 'br': self.br,
            'Wh': self.Wh, 'Uh': self.Uh, 'bh': self.bh,
            'Wo': self.Wo, 'bo': self.bo
        }

    def set_weights(self, weights_dict):
        self.Wz = weights_dict['Wz']
        self.Uz = weights_dict['Uz']
        self.bz = weights_dict['bz']
        self.Wr = weights_dict['Wr']
        self.Ur = weights_dict['Ur']
        self.br = weights_dict['br']
        self.Wh = weights_dict['Wh']
        self.Uh = weights_dict['Uh']
        self.bh = weights_dict['bh']
        self.Wo = weights_dict['Wo']
        self.bo = weights_dict['bo']

    def set_training_mode(self, is_training):
        self.is_training = is_training