from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class Grad(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="myGrad"):
        super(Grad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        pass

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        var_update = state_ops.assign_sub(var, lr_t * grad)

        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class Mom(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, bias_correction=True, use_locking=False, name="myMom"):
        super(Mom, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self.bias_correction = bias_correction

        self._lr_t = None
        self._beta1_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")

    def _create_slots(self, var_list):

        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + grad * (1.0 - beta1_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * m_t, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)

class Adam(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAdam"):
        super(Adam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        b2p = self.get_slot(var, "b2p")

        cond = tf.abs(tf.sign(grad))
        m_t = state_ops.assign(m, (beta1_t * m + grad * (1.0 - beta1_t)) * cond + m * (1.0 - cond), use_locking=self._use_locking)
        v_t = state_ops.assign(v, (beta2_t * v + (grad * grad) * (1.0 - beta2_t)) * cond + v * (1.0 - cond), use_locking=self._use_locking)
        b2p_t = state_ops.assign(b2p, b2p * beta2_t * cond + (1.0 - cond), use_locking=self._use_locking)
        b2_fix = tf.maximum(1.0 - self._beta2, 1.0 - b2p_t)

        var_update = state_ops.assign_sub(var, lr_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t) * cond, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)

class AdamMax(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdamMax"):
        super(AdamMax, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        cond = tf.abs(tf.sign(grad))
        m_t = state_ops.assign(m, (beta1_t * m + grad * (1.0 - beta1_t)) * cond + m * (1.0 - cond), use_locking=self._use_locking)
        v_t = state_ops.assign(v, (tf.maximum(beta2_t * v, grad * grad)) * cond + v * (1.0 - cond), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * m_t / (math_ops.sqrt(v_t) + epsilon_t) * cond, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)

class AdamTimeShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdamTimeShift"):
        super(AdamTimeShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "g", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        b2p = self.get_slot(var, "b2p")

        cond1 = tf.abs(tf.sign(grad))
        m_t = state_ops.assign(m, (beta1_t * m + grad * (1.0 - beta1_t)) * cond1 + m * (1.0 - cond1), use_locking=self._use_locking)

        cond2 = tf.abs(tf.sign(g))
        v_t = state_ops.assign(v, (v * beta2_t + tf.square(g) * (1.0 - beta2_t)) * cond2 + v * (1.0 - cond2), use_locking=self._use_locking)
        b2p_t = state_ops.assign(b2p, b2p * beta2_t * cond2 + (1.0 - cond2), use_locking=self._use_locking)
        b2_fix = tf.maximum(1.0 - self._beta2, 1.0 - b2p_t)

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g, grad, use_locking=self._use_locking)

        step_t = m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t) * cond2 * cond1

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, g_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)

class AdamSpaceShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdamSpaceShift"):
        super(AdamSpaceShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "g", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        b2p = self.get_slot(var, "b2p")

        cond1 = tf.abs(tf.sign(grad))
        m_t = state_ops.assign(m, (beta1_t * m + grad * (1.0 - beta1_t)) * cond1 + m * (1.0 - cond1), use_locking=self._use_locking)

        # g_square = tf.square(g)
        # def mean(g_square):
        #     return (tf.reduce_sum(g_square) - g_square) / (tf.reduce_prod(tf.shape(g_square))-1.0)
        #
        # def max(g_square):
        #     max_g_square = tf.reduce_max(g_square)
        #     cond = (g_square == max_g_square)
        #     max1_g_square = tf.reduce_max(g_square - cond * g_square)
        #     max_g_square = max_g_square * (1.0 - cond) + max1_g_square * cond
        #     return max_g_square
        #
        # gs = max(g_square)
        # gs = mean(g_square)

        gs = tf.maximum(tf.reduce_mean(tf.square(g)), tf.square(g))

        cond2 = tf.abs(tf.sign(gs))
        v_t = state_ops.assign(v, (v * beta2_t + gs * (1.0 - beta2_t)) * cond2 + v * (1.0 - cond2), use_locking=self._use_locking)
        b2p_t = state_ops.assign(b2p, b2p * beta2_t * cond2 + (1.0 - cond2), use_locking=self._use_locking)
        b2_fix = tf.maximum(1.0 - self._beta2, 1.0 - b2p_t)

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g, grad, use_locking=self._use_locking)

        step_t = m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t) * cond2 * cond1

        if 'discriminator67345715' in var.name:
            step_t = tf.Print(step_t, [cond1[0]], var.name + ' cond1:')
            step_t = tf.Print(step_t, [cond2[0]], var.name + ' cond2:')
            step_t = tf.Print(step_t, [b2_fix[0]], var.name + ' b2_fix:')
            step_t = tf.Print(step_t, [grad[0]], var.name + ' grad:')

            step_t = tf.Print(step_t, [m_t[0]], var.name + ' m_t:')
            step_t = tf.Print(step_t, [math_ops.sqrt(v_t / b2_fix)[0]], var.name + ' v_t_fix:')
            step_t = tf.Print(step_t, [step_t[0]], var.name + ' step_t:')
            step_t = tf.Print(step_t, [tf.reduce_max(step_t)], var.name + ' max_step_t:')

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, g_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)

class AdamShiftN(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, keep_num=10, beta1=0.9, beta2=0.999, epsilon=1e-10, pre_g_op='none', use_locking=False, name="AdamShiftN"):
        super(AdamShiftN, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._keep_num = keep_num
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._pre_g_op = pre_g_op

        self.s = [(self._beta1 ** (self._keep_num - i - 1)) / (1 - self._beta1 ** self._keep_num) * (1 - self._beta1) for i in range(self._keep_num)]

        self._lr_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        self.first_var = min(var_list, key=lambda x: x.name)

        for v in var_list:
            for i in range(self._keep_num+1):
                self._zeros_slot(v, "g%d" % i, self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "z", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        g = [self.get_slot(var, "g%d" % i) for i in range(self._keep_num+1)]
        v = self.get_slot(var, "v")
        z = self.get_slot(var, "z")
        b2p = self.get_slot(var, "b2p")

        if self._pre_g_op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g[0]) * (1.0 - beta2_t), use_locking=self._use_locking)
        elif self._pre_g_op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g[0])) * (1.0 - beta2_t), use_locking=self._use_locking)
        elif self._pre_g_op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g[0])) * (1.0 - beta2_t), use_locking=self._use_locking)
        else:
            assert False

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g[-1], grad, use_locking=self._use_locking)
            for i in range(self._keep_num):
                with ops.control_dependencies([g_t]):
                    g_t = state_ops.assign(g[i], g[i + 1], use_locking=self._use_locking)

        with ops.control_dependencies([g_t]):
            m_t = tf.reduce_sum([g[i] * self.s[i] for i in range(self._keep_num)], axis=0)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))

        cond = tf.sign(z_t)
        b2p_t = state_ops.assign(b2p, b2p * beta2_t * cond + (1.0 - cond), use_locking=self._use_locking)
        b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)

        step_t = z_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*([var_update]))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise NotImplementedError()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)