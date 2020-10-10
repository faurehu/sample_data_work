import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ptitprince as pt


def dist_splits(key, data):
    data = data.assign(joiner=0)
    fig = plt.figure(figsize=[15, 5])
    gs = mpl.gridspec.GridSpec(2, 7, figure=fig)
    ax0 = fig.add_subplot(gs[:, 0:3])
    ax1 = fig.add_subplot(gs[:, 3:5])
    ax2 = fig.add_subplot(gs[0, 5:7])
    ax3 = fig.add_subplot(gs[1, 5:7])

    pt.RainCloud(x='strategy', y=key, data=data, bw=.2, width_viol=.6, ax=ax0,
                 orient='v', alpha=0.5, palette='Set1', hue='n_parties', dodge=True)
    ax0.set_title('Split by n_parties')
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend().remove()
    fig.legend((handles[0], handles[1]), (labels[0],
                                          labels[1]), title='n_parties', loc=(0.9, 0.325))
    ax0.set_xlabel('')

    pt.RainCloud(x='n_parties', y=key, data=data, bw=.2, width_viol=.6, ax=ax1,
                 orient='v', alpha=0.5, palette='Set2', hue='strategy', dodge=True)
    ax1.set_title('Split by strategy')
    ax1.axes.get_yaxis().set_visible(False)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend().remove()
    fig.legend((handles[0], handles[1], handles[2]), (labels[0],
                                                      labels[1], labels[2]), title='strategy', loc=(0.9, 0.71))
    ax1.set_xlabel('')

    pt.RainCloud(x='joiner', y=key, data=data, bw=.2, width_viol=.6, ax=ax2,
                 orient='h', alpha=0.5, palette='Set2', hue='strategy', dodge=True)
    ax2.legend().remove()
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_xlabel('')

    pt.RainCloud(x='joiner', y=key, data=data, bw=.2, width_viol=.6, ax=ax3,
                 orient='h', alpha=0.5, palette='Set1', hue='n_parties', dodge=True)
    ax3.legend().remove()
    ax3.axes.get_yaxis().set_visible(False)
    ax3.set_xlabel('')

    fig.suptitle('{0} distributions'.format(key), fontsize=14)
    plt.subplots_adjust(wspace=0.08, hspace=0.08)
    return fig


def _predict(p, qval, lval, kval, aggv, nparties3v):
    return p.Intercept + (p.q * qval) + (p.lambd * lval) + (p.kappa * kval) + (p.nparties3 * nparties3v) + (p.aggregator * aggv) + (p['aggregator:q'] * aggv * qval) + (p['np.power(q, 2)'] * qval ** 2) + (p['np.power(lambd, 2)'] * lval ** 2) + (p['lambd:nparties3'] * nparties3v * lval) + (p['np.power(lambd, 2):nparties3'] * nparties3v * lval ** 2) + (p['lambd:aggregator'] * lval * aggv) + (p['q:nparties3'] * qval * nparties3v) + (p['lambd:q'] * lval * qval)


def plot_predict(res, data):
    agg_c = 'darkred'
    nagg_c = 'darkgreen'
    twopstyle = '--'
    threepstyle = ':'

    p = res.params
    qsupport = np.arange(0, 40)
    lsupport = np.arange(0, 1, 0.001)
    ksupport = np.arange(0, 1, 0.001)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    plt.rc('lines', linewidth=3)

    axs[0].scatter(data.q, data.mean_loss, alpha=0.3)
    # nonaggregator 2p
    axs[0].plot(qsupport, _predict(p, qsupport, 0.5, 1, 0, 0),
                color=nagg_c, linestyle=twopstyle)
    # nonaggregator 3p
    axs[0].plot(qsupport, _predict(p, qsupport, 0.5, 1, 0, 1),
                color=nagg_c, linestyle=threepstyle)
    # aggregator 2p
    axs[0].plot(qsupport, _predict(p, qsupport, 0.5, 1, 1, 0),
                color=agg_c, linestyle=twopstyle)
    # aggregator 3p
    axs[0].plot(qsupport, _predict(p, qsupport, 0.5, 1, 1, 1),
                color=agg_c, linestyle=threepstyle)
    axs[0].set_ylabel('Mean loss')
    axs[0].set_xlabel('q')

    axs[1].scatter(data.lambd, data.mean_loss, alpha=0.3)
    # sticker / hunter 2p
    axs[1].plot(lsupport, _predict(p, 10, lsupport, 1, 0, 0),
                color=nagg_c, linestyle=twopstyle)
    # sticker / hunter 3p
    axs[1].plot(lsupport, _predict(p, 10, lsupport, 1, 0, 1),
                color=nagg_c, linestyle=threepstyle)
    # aggregator 2p
    axs[1].plot(lsupport, _predict(p, 10, lsupport, 1, 1, 0),
                color=agg_c, linestyle=twopstyle)
    # aggregator 3p
    axs[1].plot(lsupport, _predict(p, 10, lsupport, 1, 1, 1),
                color=agg_c, linestyle=threepstyle)
    axs[1].set_xlabel('lambda')
    axs[1].legend(['2 non-aggregators', '3 non-aggregators',
                   '2 aggregators', '3 aggregators'])

    axs[2].scatter(data.kappa, data.mean_loss, alpha=0.3)
    # sticker / hunter 2p
    axs[2].plot(ksupport, _predict(p, 10, 0.5, ksupport, 0, 0),
                color=nagg_c, linestyle=twopstyle)
    # sticker /hunter 3p
    axs[2].plot(ksupport, _predict(p, 10, 0.5, ksupport, 0, 1),
                color=nagg_c, linestyle=threepstyle)
    # agg 2p
    axs[2].plot(ksupport, _predict(p, 10, 0.5, ksupport, 1, 0),
                color=agg_c, linestyle=twopstyle)
    axs[2].plot(ksupport, _predict(p, 10, 0.5, ksupport, 1, 1),
                color=agg_c, linestyle=threepstyle)
    axs[2].set_xlabel('theta')

    return fig


def _zfunc(a, b):
    return (a ** 2 + b ** 2)/a


def plot_surface():
    x = np.arange(0.01, 0.2, 0.01)
    y = np.arange(-0.2, 0.2, 0.01)
    X, Y = np.meshgrid(x, y)
    z = np.array([_zfunc(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = z.reshape(X.shape)

    nx = np.arange(-0.2, -0.01, 0.01)
    nX, nY = np.meshgrid(nx, y)
    nz = np.array([_zfunc(x, y) for x, y in zip(np.ravel(nX), np.ravel(nY))])
    nZ = nz.reshape(nX.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X, Y, Z, linewidth=0, cmap='viridis')
    ax.set_xlabel("dp")
    ax.set_xticks([0.00, .05, 0.10, 0.15, 0.20])
    ax.set_ylabel("dq")
    ax.set_zlabel("f(dp,dq)")

    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(nX, nY, nZ, linewidth=0, cmap='viridis')
    ax1.set_xlabel("dp")
    ax1.set_xticks([0.00, -0.05, -0.1, -0.15, -0.20])
    ax1.set_ylabel("dq")
    ax1.set_zlabel("f(dp,dq)")

    return fig


def plot_manifesto_2d(cpl):
    fig, ax = plt.subplots(figsize=(8, 8))
    countries = cpl.countryname.unique()

    ax.set_title('Manifesto Content')
    ax.set_ylabel('Average number of proposals in manifesto')
    ax.set_xlabel('Year')

    for i in countries:
        sub = cpl[cpl.countryname == i]
        ax.plot(sub.edate, sub.total, linestyle='solid',
                color='black', alpha=0.1)

    return fig
