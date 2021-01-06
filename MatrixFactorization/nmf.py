import numpy as np
import nimfa
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import check_non_negative


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'kullback-leibler': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _check_string_param(solver, regularization, beta_loss, init):
    allowed_solver = ('cd', 'mu')
    if solver not in allowed_solver:
        raise ValueError(
            'Invalid solver parameter: got %r instead of one of %r' %
            (solver, allowed_solver))

    allowed_regularization = ('both', 'components', 'transformation', None)
    if regularization not in allowed_regularization:
        raise ValueError(
            'Invalid regularization parameter: got %r instead of one of %r' %
            (regularization, allowed_regularization))

    # 'mu' is the only solver that handles other beta losses than 'frobenius'
    if solver != 'mu' and beta_loss not in (2, 'frobenius'):
        raise ValueError(
            'Invalid beta_loss parameter: solver %r does not handle beta_loss'
            ' = %r' % (solver, beta_loss))

    if solver == 'mu' and init == 'nndsvd':
        warnings.warn("The multiplicative update ('mu') solver cannot update "
                      "zeros present in the initialization, and so leads to "
                      "poorer results when used jointly with init='nndsvd'. "
                      "You may try init='nndsvda' or init='nndsvdar' instead.",
                      UserWarning)

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss




def non_negative_factorization(X, W=None, H=None, n_components=None,
                               init='warn', update_H=True, solver='cd',
                               beta_loss='frobenius', tol=1e-4,
                               max_iter=200, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=None,
                               verbose=0, shuffle=False):

    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if X.min() == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, numbers.Integral) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, numbers.Integral) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    if init == "warn":
        if n_components < n_features:
            warnings.warn("The default value of init will change from "
                          "random to None in 0.23 to make it consistent "
                          "with decomposition.NMF.", FutureWarning)
        init = "random"

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        # 'mu' solver should not be initialized by zeros
        if solver == 'mu':
            avg = np.sqrt(X.mean() / n_components)
            W = np.full((n_samples, n_components), avg)
        else:
            W = np.zeros((n_samples, n_components))
    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    if solver == 'cd':
        W, H, n_iter = _fit_coordinate_descent(X, W, H, tol, max_iter,
                                               l1_reg_W, l1_reg_H,
                                               l2_reg_W, l2_reg_H,
                                               update_H=update_H,
                                               verbose=verbose,
                                               shuffle=shuffle,
                                               random_state=random_state)
    elif solver == 'mu':
        W, H, n_iter = _fit_multiplicative_update(X, W, H, beta_loss, max_iter,
                                                  tol, l1_reg_W, l1_reg_H,
                                                  l2_reg_W, l2_reg_H, update_H,
                                                  verbose)

    else:
        raise ValueError("Invalid solver parameter '%s'." % solver)

    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter









def nmf(X,W,H):

    of = abs(X-WH)**2
0.5 * ||X - WH||_Fro^2
+ alpha * l1_ratio * ||vec(W)||_1
+ alpha * l1_ratio * ||vec(H)||_1
+ 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2
+ 0.5 * alpha * (1 - l1_ratio) * ||H||_Fro^2

>>> import numpy as np
>>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
>>> from sklearn.decomposition import NMF
>>> model = NMF(n_components=2, init='random', random_state=0)
>>> W = model.fit_transform(X)
>>> H = model.components_

