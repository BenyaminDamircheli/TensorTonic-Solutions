import numpy as np

def _momentum(beta1, m, grad):
    return beta1 * m + (1-beta1) * grad

def _adaptive_rate(beta2, v, grad):
    return beta2 * v + (1-beta2) * grad**2

def update(x, lr, m_t, v_t, eps):
    return x - lr * (m_t)/(np.sqrt(v_t) + eps)

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """

    param = np.array(param)
    grad = np.array(grad)
    m = np.array(m)
    v = np.array(v)

    momentum = _momentum(beta1, m, grad)
    adaptive_rate = _adaptive_rate(beta2, v, grad)
    m_new = momentum / (1 - beta1**t)
    v_new = adaptive_rate / (1 - beta2**t)

    param_new = update(param, lr, m_new, v_new, eps)

    return (param_new, momentum, adaptive_rate)
    
