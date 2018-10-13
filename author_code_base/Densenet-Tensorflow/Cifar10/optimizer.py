from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np


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

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1.0 - beta1_power if self.bias_correction else 1.0

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


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

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AMSGrad(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, tf.maximum(beta2_t * v + (grad * grad) * (1 - beta2_t), v), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


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

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, tf.maximum(beta2_t * v, grad * grad), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, grad * (1 - beta1_t))

        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, (grad * grad) * (1 - beta2_t))

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t) + epsilon_t), use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


# class AdamMaxScale(optimizer.Optimizer):
#
#     def __init__(self, learning_rate=0.001, alpha=0.01, beta1=0.9, beta2=0.999, beta3=0.9999, epsilon=1e-10, use_locking=False, name="AdamMaxScale"):
#         super(AdamMaxScale, self).__init__(use_locking, name)
#         self._lr = learning_rate
#         self._alpha = alpha
#         self._beta1 = beta1
#         self._beta2 = beta2
#         self._beta3 = beta3
#         self._epsilon = epsilon
#
#         self._lr_t = None
#         self._beta1_t = None
#         self._beta2_t = None
#         self._beta3_t = None
#         self._epsilon_t = None
#
#         self._beta1_power = None
#         self._beta2_power = None
#         self._beta3_power = None
#
#     def _prepare(self):
#         self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
#         self._alpha_t = ops.convert_to_tensor(self._alpha, name="alpha")
#         self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
#         self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
#         self._beta3_t = ops.convert_to_tensor(self._beta3, name="beta3")
#         self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")
#
#     def _create_slots(self, var_list):
#
#         first_var = min(var_list, key=lambda x: x.name)
#
#         create_new = self._beta1_power is None
#         if not create_new and tf.context.in_graph_mode():
#             create_new = (self._beta1_power.graph is not first_var.graph)
#
#         if create_new:
#             with ops.colocate_with(first_var):
#                 self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
#                 self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)
#                 self._beta3_power = tf.Variable(self._beta2, name="beta3_power", trainable=False)
#
#         for v in var_list:
#             self._zeros_slot(v, "m", self._name)
#             self._zeros_slot(v, "v", self._name)
#             self._zeros_slot(v, "w", self._name)
#
#     def _apply_dense(self, grad, var):
#
#         lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
#         alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
#         beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
#         beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
#         beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
#         epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
#
#         beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
#         beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
#         beta3_power = math_ops.cast(self._beta3_power, var.dtype.base_dtype)
#
#         beta1_fix = 1 - beta1_power
#         beta2_fix = 1 - beta2_power
#         beta3_fix = 1 - beta3_power
#
#         m = self.get_slot(var, "m")
#         v = self.get_slot(var, "v")
#         w = self.get_slot(var, "w")
#
#         m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
#         v_t = state_ops.assign(v, tf.maximum(beta2_t * v, grad * grad), use_locking=self._use_locking) # Aad-Max
#         # v_t = state_ops.assign(v, tf.maximum(beta2_t * v + (grad * grad) * (1 - beta2_t), v), use_locking=self._use_locking)  # AMSGrad
#         # v_t = state_ops.assign(v, tf.maximum(beta2_t * v + (grad * grad) * (1 - beta2_t), v * beta3_t), use_locking=self._use_locking)
#         w_t = state_ops.assign(w, beta3_t * w + tf.abs(var) * (1 - beta3_t), use_locking=self._use_locking)
#
#         var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t) * (1.0 + (w_t / beta3_fix) / alpha_t), use_locking=self._use_locking)
#         return control_flow_ops.group(*[var_update, m_t, v_t, w_t])
#
#     def _apply_sparse_shared(self, grad, var, indices, scatter_add):
#
#         lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
#         alpha_t = math_ops.cast(self._alpha_t, var.dtype.base_dtype)
#         beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
#         beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
#         beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
#         epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
#
#         beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
#         beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
#         beta3_power = math_ops.cast(self._beta3_power, var.dtype.base_dtype)
#
#         beta1_fix = 1 - beta1_power
#         beta2_fix = 1 - beta2_power
#         beta3_fix = 1 - beta3_power
#
#         m = self.get_slot(var, "m")
#         m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
#         with ops.control_dependencies([m_t]):
#             m_t = scatter_add(m, indices, grad * (1 - beta1_t))
#
#         v = self.get_slot(var, "v")
#         v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
#         with ops.control_dependencies([v_t]):
#             v_t = scatter_add(v, indices, (grad * grad) * (1 - beta2_t))
#
#         w = self.get_slot(var, "w")
#         w_t = state_ops.assign(w, beta3_t * w + tf.abs(var) * (1 - beta3_t), use_locking=self._use_locking)
#
#         var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t) * (1.0 + (w_t / beta3_fix) / alpha_t), use_locking=self._use_locking)
#
#         return control_flow_ops.group(*[var_update, m_t, v_t, w_t])
#
#     def _apply_sparse(self, grad, var):
#         return self._apply_sparse_shared(
#             grad.values, var, grad.indices,
#             lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))
#
#     def _finish(self, update_ops, name_scope):
#         with ops.control_dependencies(update_ops):
#           with ops.colocate_with(self._beta1_power):
#             update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
#             update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
#             update_beta3 = self._beta3_power.assign(self._beta3_power * self._beta3_t, use_locking=self._use_locking)
#         return control_flow_ops.group(*update_ops + [update_beta1, update_beta2, update_beta3], name=name_scope)


class AdamShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, pred_g_op='none', use_locking=False, name="AdamShift"):
        super(AdamShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._pred_g_op = pred_g_op

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

        self.first_var = min(var_list, key=lambda x: x.name)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "g", self._name)
            self._zeros_slot(v, "z", self._name)
            self._zeros_slot(v, "b1p", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")
        z = self.get_slot(var, "z")
        b1p = self.get_slot(var, "b1p")
        b2p = self.get_slot(var, "b2p")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)

        if self._pred_g_op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g)) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g)) * (1 - beta2_t), use_locking=self._use_locking)
        else:
            assert False

        # v_t = tf.cond(tf.less(self._current_iter, tf.constant(self._init_step)),
        #               lambda: state_ops.assign(v, v * beta2_t + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking),
        #               lambda: state_ops.assign(v, v * beta2_t + (g * g) * (1 - beta2_t), use_locking=self._use_locking))

        # cond = (tf.sign(tf.cast(self._current_iter - tf.constant(self._init_step), tf.float32) + tf.constant(0.5)) + tf.constant(1.0)) / tf.constant(2.0)
        # v_a = v * beta2_t + (grad * grad) * (1 - beta2_t)
        # v_b = v * beta2_t + (g * g) * (1 - beta2_t)
        # v_t = state_ops.assign(v, v_a * (1 - cond) + v_b * cond, use_locking=self._use_locking)

        # cond = tf.abs(tf.sign(g))
        # v_t = state_ops.assign(v, v * (1 - cond) + (v * beta2_t + (g * g) * (1 - beta2_t)) * cond, use_locking=self._use_locking)

        # v_t = state_ops.assign(v, v * beta2_t + (g * g) * (1 - beta2_t), use_locking=self._use_locking)
        # v_t = state_ops.assign(v, tf.maximum(grad * grad * beta2_fix, v * beta2_t + (g * g) * (1 - beta2_t)), use_locking=self._use_locking)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))
            g_t = state_ops.assign(g, grad, use_locking=self._use_locking)

        b1p_t = state_ops.assign(b1p, b1p * beta1_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)
        b2p_t = state_ops.assign(b2p, b2p * beta2_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)

        b1_fix = tf.maximum(1e-8, 1.0 - b1p_t)
        b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)

        step_t = z_t * (m_t / b1_fix) / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        # if var.name == self.first_var.name: #'discriminator/final_linear/w:0':
        #     idx = 0
        #     step_t = tf.Print(step_t, [z_t[idx]], 'z_t', summarize=1000)
        #     step_t = tf.Print(step_t, [g[idx]], 'g', summarize=1000)
        #     step_t = tf.Print(step_t, [grad[idx]], 'grad', summarize=1000)
        #     step_t = tf.Print(step_t, [b2p_t[idx]], 'b2p_t', summarize=1000)
        #     step_t = tf.Print(step_t, [b2_fix], 'beta2_fix', summarize=1000)
        #     step_t = tf.Print(step_t, [tf.sqrt(v_t / b2_fix)[idx]], 'v_t', summarize=1000)
        #     step_t = tf.Print(step_t, [step_t], 'step', summarize=1000)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, g_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise Exception()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)


class AdamShiftN(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, keep_num=10, beta2=0.999, epsilon=1e-10, pred_g_op='none', use_locking=False, name="AdamShiftN"):
        super(AdamShiftN, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._keep_num = keep_num
        self._beta2 = beta2
        self._epsilon = epsilon
        self._pred_g_op = pred_g_op

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

        if self._pred_g_op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g[0]) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        else:
            assert False

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g[-1], grad, use_locking=self._use_locking)
            for i in range(self._keep_num):
                with ops.control_dependencies([g_t]):
                    g_t = state_ops.assign(g[i], g[i + 1], use_locking=self._use_locking)

        with ops.control_dependencies([g_t]):
            m_t = tf.reduce_mean(g[:self._keep_num], axis=0)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))

        b2p_t = state_ops.assign(b2p, b2p * beta2_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)
        b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)

        step_t = z_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        # if var.name == self.first_var.name: #'discriminator/final_linear/w:0':
        #     idx = 0
        #     step_t = tf.Print(step_t, [z_t[idx]], 'z_t', summarize=1000)
        #     step_t = tf.Print(step_t, [g[i][idx] for i in range(len(g))], 'g', summarize=1000)
        #     step_t = tf.Print(step_t, [grad[idx]], 'grad', summarize=1000)
        #     step_t = tf.Print(step_t, [b2p_t[idx]], 'b2p_t', summarize=1000)
        #     step_t = tf.Print(step_t, [b2_fix], 'beta2_fix', summarize=1000)
        #     step_t = tf.Print(step_t, [m_t[idx]], 'm_t', summarize=1000)
        #     step_t = tf.Print(step_t, [tf.sqrt(v_t / b2_fix)[idx]], 'v_t', summarize=1000)
        #     step_t = tf.Print(step_t, [step_t], 'step', summarize=1000)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*([var_update]))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise Exception()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)


class AdamShiftMoving(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, keep_num=10, beta1=0.9, beta2=0.999, epsilon=1e-10, pred_g_op='none', use_locking=False, name="AdamShiftMoving"):
        super(AdamShiftMoving, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._keep_num = keep_num
        self._beta2 = beta2
        self._beta1 = beta1
        self._epsilon = epsilon
        self._pred_g_op = pred_g_op

        s = np.asarray([(self._beta1**(self._keep_num-i-1)) for i in range(self._keep_num)])
        self.s = s / np.sum(s) 
        
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

        if self._pred_g_op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g[0]) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        else:
            assert False

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g[-1], grad, use_locking=self._use_locking)
            for i in range(self._keep_num):
                with ops.control_dependencies([g_t]):
                    g_t = state_ops.assign(g[i], g[i + 1], use_locking=self._use_locking)

        with ops.control_dependencies([g_t]):
            m_t = tf.reduce_sum([g[i]*self.s[i] for i in range(self._keep_num)], axis=0)
            # m_t = tf.reduce_mean(g[:self._keep_num], axis=0)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))

        b2p_t = state_ops.assign(b2p, b2p * beta2_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)
        b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)

        step_t = z_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        # if var.name == self.first_var.name: #'discriminator/final_linear/w:0':
        #     idx = 0
        #     step_t = tf.Print(step_t, [z_t[idx]], 'z_t', summarize=1000)
        #     step_t = tf.Print(step_t, [g[i][idx] for i in range(len(g))], 'g', summarize=1000)
        #     step_t = tf.Print(step_t, [grad[idx]], 'grad', summarize=1000)
        #     step_t = tf.Print(step_t, [b2p_t[idx]], 'b2p_t', summarize=1000)
        #     step_t = tf.Print(step_t, [b2_fix], 'beta2_fix', summarize=1000)
        #     step_t = tf.Print(step_t, [m_t[idx]], 'm_t', summarize=1000)
        #     step_t = tf.Print(step_t, [tf.sqrt(v_t / b2_fix)[idx]], 'v_t', summarize=1000)
        #     step_t = tf.Print(step_t, [step_t], 'step', summarize=1000)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*([var_update]))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise Exception()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)