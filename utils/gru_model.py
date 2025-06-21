import numpy as np
from .activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative

class GRU:
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.is_training = True

        # Inisialisasi bobot (weights)
        # Bentuk: (input_dim, hidden_dim) untuk bobot input
        # Bentuk: (hidden_dim, hidden_dim) untuk bobot hidden state
        # Bentuk: (1, hidden_dim) untuk bias
        # Menggunakan np.random.randn * 0.01 untuk inisialisasi kecil
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
        Melakukan forward pass pada GRU untuk satu batch sequences.

        Args:
            X (np.ndarray): Input batch, shape (batch_size, sequence_length, input_dim)
        Returns:
            np.ndarray: Output prediksi untuk setiap sequence di batch, shape (batch_size, output_dim)
        """
        batch_size, sequence_length, _ = X.shape

        # Inisialisasi hidden state awal untuk setiap sampel di batch
        h_prev = np.zeros((batch_size, self.hidden_dim))

        # Cache untuk menyimpan nilai-nilai per time step untuk backward pass
        sequence_cache = []

        for t in range(sequence_length):
            x_t = X[:, t, :] 

            # Perhitungan Update Gate (z_t)
            # Wz: (input_dim, hidden_dim), x_t: (batch_size, input_dim) -> (batch_size, hidden_dim)
            # Uz: (hidden_dim, hidden_dim), h_prev: (batch_size, hidden_dim) -> (batch_size, hidden_dim)
            z_pre = x_t @ self.Wz + h_prev @ self.Uz + self.bz # bz akan di-broadcast
            z_t = sigmoid(z_pre)

            # Perhitungan Reset Gate (r_t)
            r_pre = x_t @ self.Wr + h_prev @ self.Ur + self.br
            r_t = sigmoid(r_pre)

            # Perhitungan Candidate Hidden State (h_tilde_t)
            h_tilde_pre = x_t @ self.Wh + (r_t * h_prev) @ self.Uh + self.bh
            h_tilde_t = tanh(h_tilde_pre)

            # Perhitungan Final Hidden State (h_t)
            h_t = (1 - z_t) * h_prev + z_t * h_tilde_t

            # Dropout (inverted dropout)
            dropout_mask = np.ones_like(h_t)
            if self.is_training and self.dropout_rate > 0:
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

        # Output lapisan akhir (linear layer) hanya dari hidden state terakhir
        final_h_t = h_t 
        output = final_h_t @ self.Wo + self.bo 

        self.cache['sequence_cache'] = sequence_cache
        self.cache['final_h_t'] = final_h_t 
        self.cache['output'] = output
        return output

    def backward(self, d_output):
        """
        Melakukan backward pass untuk satu batch sequences.

        Args:
            d_output (np.ndarray): Gradien dari loss terhadap output model, shape (batch_size, output_dim)
        """
        sequence_cache = self.cache['sequence_cache']

        # Inisialisasi gradien untuk bobot
        # Gradien akan diakumulasikan di seluruh time steps dan batch samples
        dWz, dUz, dbz = np.zeros_like(self.Wz), np.zeros_like(self.Uz), np.zeros_like(self.bz)
        dWr, dUr, dbr = np.zeros_like(self.Wr), np.zeros_like(self.Ur), np.zeros_like(self.br)
        dWh, dUh, dbh = np.zeros_like(self.Wh), np.zeros_like(self.Uh), np.zeros_like(self.bh)
        dWo, dbo = np.zeros_like(self.Wo), np.zeros_like(self.bo)

        # Gradien dari output layer
        dWo += sequence_cache[-1]['h_t'].T @ d_output 
        dbo += np.sum(d_output, axis=0, keepdims=True) 

        # Inisialisasi gradien hidden state dari output layer
        dh_next = d_output @ self.Wo.T 

        for t in reversed(range(len(sequence_cache))):
            cache_t = sequence_cache[t]
            x_t, h_prev, z_pre, r_pre, h_tilde_pre, z_t, r_t, h_tilde_t, h_t_cached, dropout_mask = \
                cache_t['x_t'], cache_t['h_prev'], cache_t['z_pre'], cache_t['r_pre'], cache_t['h_tilde_pre'], \
                cache_t['z_t'], cache_t['r_t'], cache_t['h_tilde_t'], cache_t['h_t'], cache_t['dropout_mask']

            dh_t = dh_next * dropout_mask 

            dh_prev_term1 = dh_t * (1 - z_t)

            dz_t = dh_t * (h_tilde_t - h_prev)
            dz_pre = dz_t * sigmoid_derivative(z_pre)

            dh_tilde_t = dh_t * z_t
            dh_tilde_pre = dh_tilde_t * tanh_derivative(h_tilde_pre)

            dWh += x_t.T @ dh_tilde_pre
            dbh += np.sum(dh_tilde_pre, axis=0, keepdims=True)

            dUh += (r_t * h_prev).T @ dh_tilde_pre

            dr_t_from_h_tilde = dh_tilde_pre @ self.Uh.T * h_prev

            dr_t = dr_t_from_h_tilde
            dr_pre = dr_t * sigmoid_derivative(r_pre)

            dWr += x_t.T @ dr_pre
            dbr += np.sum(dr_pre, axis=0, keepdims=True)
            dUr += h_prev.T @ dr_pre

            dWz += x_t.T @ dz_pre
            dbz += np.sum(dz_pre, axis=0, keepdims=True)
            dUz += h_prev.T @ dz_pre

            dh_next = dh_prev_term1 + \
                      (dz_pre @ self.Uz.T) + \
                      (dr_pre @ self.Ur.T) + \
                      (dh_tilde_pre @ self.Uh.T) * r_t 

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