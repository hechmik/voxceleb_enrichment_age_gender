from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer


class Padam(Optimizer):
    def __init__(self, lr=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, decay=0., amsgrad=False,
                 partial=0.125, **kwargs):
        """ Partially adaptive momentum estimation method (Padam).
        # Arguments
            lr: float >= 0. Learning rate.
            beta_1: float, 0 < beta < 1. Generally close to 1.
            beta_2: float, 0 < beta < 1. Generally close to 1.
            epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.
            decay: float >= 0. Learning rate decay over each update.
            amsgrad: boolean. Whether to apply the AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                Beyond".
            partial: float >=0 and <= 0.5. Partially adaptive parameter, with a maximum
                value of 0.5. Values larger than 0.5 will result in non convergence.
                As `partial` tends to 0, Padam reduces to SGD, and as `partial` tends
                to 0.5, Padam reduces to AMSGrad. Default value is obtained empirically.
        # References
            - [Closing the Generalization Gap of Adaptive Gradient Methods in Training Deep Neural Networks](https://arxiv.org/abs/1806.06763)
            - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
        """
        super(Padam, self).__init__(**kwargs)

        if partial < 0.0 or partial > 0.5:
            raise ValueError("`partial` must be a positive float between 0 and 0.5, "
                             "otherwise the optimizer may not converge.")

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.partial = partial

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                denom = (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                denom = (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            # Partial momentum adaption
            new_p = p - (lr_t * (m_t / (denom ** (self.partial * 2))))

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad,
                  'partial': self.partial}

        base_config = super(Padam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))