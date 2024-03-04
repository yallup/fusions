import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples, make_2d_axes
from anesthetic.utils import compress_weights, neff

# from anesthetic.read.hdf import read_hdf, write_hdf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm

from fusions.model import Model
from fusions.utils import unit_hyperball, unit_hypercube


@dataclass
class Point:
    x: np.ndarray
    logl: float
    logl_birth: float
    logl_pi: float = field(default=0.0)


@dataclass
class Stats:
    nlive: int = field(default=0)
    nlike: int = field(default=0)
    ndead: int = field(default=0)
    logz: float = field(default=-1e30)
    logz_err: float = field(default=1)
    logX: float = field(default=0)

    def __repr__(self):
        return (
            f"Stats(\n"
            f"  nlive: {self.nlive},\n"
            f"  nlike: {self.nlike},\n"
            f"  ndead: {self.ndead},\n"
            f"  logz: {self.logz},\n"
            f"  logz_err: {self.logz_err},\n"
            f"  logX: {self.logX}\n"
            f")"
        )


@dataclass
class Settings:
    n: int = 5000
    target_eff: float = 0.1
    steps: int = 20
    prior_boost: int = 5
    eps: float = 1e-3
    efficiency: float = 1 / np.e
    logzero: float = -1e30

    def __repr__(self):
        return (
            f"Settings(\n"
            f"  n: {self.n},\n"
            f"  target_eff: {self.target_eff},\n"
            f"  steps: {self.steps},\n"
            f"  prior_boost: {self.prior_boost},\n"
            f"  eps: {self.eps},\n"
            f"  efficiency: {self.efficiency},\n"
            f")"
        )


@dataclass
class Trace:
    loss: float
    live: Point


class Integrator(ABC):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -np.inf)
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.settings = Settings()
        self.model = kwargs.get("model", CFM)
        latent = kwargs.get("latent", unit_hyperball)
        self.latent = latent(prior.dim, scale=1.0)

        self.trace = {}
        # self.latent = multivariate_normal(
        #     np.zeros(prior.dim), np.eye(prior.dim)
        # )

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        # if isinstance(dist, Model):
        #     idx = compress_weights(1/np.exp(j),ncompress="equal")
        #     x, j = dist.rvs(n, solution="exact", jac = False)
        # else:
        #     x = np.asarray(dist.rvs(n))
        #     j = np.zeros(n)
        # if isinstance(dist, Model):
        #     x, j  = self.dist.rvs(n, jac=True, solution="exact")
        #     w = 1 / np.exp(j)
        # else:
        #     x = np.asarray(dist.rvs(n))
        #     w = np.ones(n)

        # x = np.asarray(dist.rvs(n))
        # log_pi = self.prior.logpdf(x)
        if isinstance(dist, Model):
            x, j = dist.rvs(n, jac=True, solution="exact")
            # w = np.exp(1/np.asarray(j))
            # dist.latent.rvs(n)
            # x, j = dist.rvs(n, jac=True, solution="none")
            log_pi = self.prior.logpdf(x)
            x = np.asarray(x)
            # w = np.ones(n)
            w = np.exp(log_pi - j)
            w = 1 / dist.predict_weight(x).squeeze()
            print(np.log(w))

        else:
            x = np.asarray(dist.rvs(n))
            w = np.ones(n)
        log_pi = self.prior.logpdf(x)

        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        idx = compress_weights(w, ncompress="equal")
        print(sum(idx))
        # idx = np.ones_like(log_pi, dtype=bool)
        points = [
            Point(x[i], logl[i], logl_birth[i], logl_pi=log_pi[i])
            for i in range(x[idx].shape[0])
        ]
        return points

    def stash(self, points, n, drop=False):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        if not drop:
            self.dead += live[n:]
        contour = live[n].logl
        live = live[:n]
        return live, contour

    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def update_stats(self, live, n):
        pass

    @abstractmethod
    def points_to_samples(self, points):
        pass

    def samples(self):
        return self.points_to_samples(self.dead)

    def importance_integrate(self, dist, n=1000):
        points = dist.rvs(n)
        likelihood = self.likelihood.logpdf(points)
        prior = self.prior.logpdf(points)
        return logsumexp(likelihood + prior) - np.log(n)

    def write(self, filename):
        # Current anesthetic IO seems cumbersome.
        raise NotImplementedError

    def read(self, filename):
        raise NotImplementedError


class NestedDiffusion(Integrator):
    # def sample_constrained(
    #     self, n, dist, constraint, efficiency=0.1, **kwargs
    # ):
    #     success = []
    #     trials = 0
    #     while len(success) < n:
    #         batch_success = []
    #         pi = self.sample(n, dist, constraint, **kwargs)
    #         batch_success += [p for p in pi if p.logl > constraint]
    #         success += batch_success
    #         trials += n
    #         if len(batch_success) < n * efficiency:
    #             return False, len(success) / trials, batch_success
    #     return True, len(success) / trials, success

    def sample_constrained(self, n, dist, constraint, efficiency=0.1, **kwargs):
        trials = int(n * 1 / efficiency)
        pi = self.sample(trials, dist, constraint, **kwargs)
        success = [p for p in pi if p.logl > constraint]
        if len(success) < n:
            return False, len(success) / trials, success
        return True, len(success) / trials, success

    def run(self, n=500, target_eff=0.1, steps=20, prior_boost=1, eps=1e-3):
        live = self.sample(n * self.settings.prior_boost, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n // 2, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        diffuser = self.model(self.latent)

        # while step < steps:
        # eps=1
        while not self.points_to_samples(live + self.dead).terminate():
            success, eff, points = self.sample_constrained(
                n // 2,
                self.dist,
                contour,
                efficiency=target_eff,
            )
            step += 1
            # live, contour = self.stash(live + points, n)

            if success or len(live) >= n:
                live, contour = self.stash(live + points, n // 2)
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
                self.update_stats(live, n)
                logger.info(f"{self.stats}")
            # step += 1
            if not success and len(live) < n:
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                # print(np.asarray([yi.x for yi in live + points]).std(axis=0).mean())
                # diffuser = Diffusion(self.prior)
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=len(live) * 10,
                    batch_size=n,
                    lr=1e-3,
                    # noise = np.asarray([yi.x for yi in live + points]).std(axis=0).mean()
                )
                live += points
                self.dists.append(diffuser)
                self.dist = diffuser
                # plt.style.use("computermodern")
                # f, a = plt.subplots(
                #     ncols=4, figsize=(11, 3), sharex=True, sharey=True
                # )
                self.dist.calibrate(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=len(live) * 5,
                    batch_size=n,
                )
                f, a = make_2d_axes(
                    np.arange(self.prior.dim), upper=False, diagonal=False
                )
                # a=MCMCSamples(self.prior.rvs(200)).plot_2d(a, kinds={"lower":"scatter_2d"})
                a = MCMCSamples([p.x for p in live + points]).plot_2d(
                    a,
                    kinds={
                        "lower": "scatter_2d"
                    },  # , lower_kwargs={"c": self.dist.predict_weight(np.asarray([p.x for p in live + points])).squeeze()}
                )
                f.savefig(f"plots/step_{step}_corner.pdf")
                # f, a = plt.subplots(
                #     ncols=4, figsize=(11, 3), sharex=True, sharey=True
                # )

                # cand, j = self.dist.rvs(len(live), jac=True, solution="exact")
                # ratio = 1 / np.log(self.dist.predict_weight(cand).squeeze())
                # weight = np.exp(self.prior.logpdf(cand))
                # # weight = j - np.exp(self.prior.logpdf(cand))
                # a[0].scatter(*cand[...,0].T, c=j, alpha=0.7, rasterized=True)
                # a[1].scatter(*cand.T, c=weight, alpha=0.7, rasterized=True)
                # a[2].scatter(*cand.T, c=ratio, alpha=0.7, rasterized=True)
                # # a[0].scatter(*MCMCSamples(cand,weights = compress_weights(1/np.exp(j))).compress().to_numpy().T)
                # a[3].scatter(
                #     *cand.T,
                #     c=(
                #         self.dist.predict_weight(cand).squeeze()
                #         * self.prior.pdf(cand)
                #     )
                #     - 1 / j,
                #     alpha=0.7,
                #     rasterized=True,
                # )
                # a[0].scatter(
                #     *np.asarray([yi.x for yi in live]).T,
                #     c="C1",
                #     alpha=1.0,
                #     marker=".",
                #     s=3,
                # )
                # a[1].scatter(
                #     *np.asarray([yi.x for yi in live]).T,
                #     c="C1",
                #     alpha=1.0,
                #     marker=".",
                #     s=3,
                # )
                # a[0].set_title("Jacobian")
                # a[1].set_title("Prior")
                # a[2].set_title("Classifier")
                # # a[0].set_xlim(-2.5, 1.5)
                # # a[0].set_ylim(-1.5, 2.0)
                # # a[1].set_xlim(-2.5, 1.5)
                # # a[1].set_ylim(-1.5, 2.0)
                # a[3].set_title(r"Classifier $\times$ Prior - Jac")
                # # a[2].set_xlim(-2.5, 1.5)
                # # a[2].set_ylim(-1.5, 2.0)
                # f.tight_layout(pad=0.5)
                # f.savefig(f"plots/step_{step}_disc.pdf")
            logger.info(f"Step {step}/{steps} complete")

        self.stash(live, -len(live))
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead + live)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        self.stats.logX = -(len(self.dead) - n * self.settings.prior_boost) / n
        self.stats.logz = lZs.mean()
        self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        return NestedSamples(
            data=[p.x for p in points],
            logL=[p.logl for p in points],
            logL_birth=[p.logl_birth for p in points],
        )

    def points_to_samples_importance(self, points, weights):
        return MCMCSamples(
            data=[p.x for p in points],
            weights=weights,
            # weights=np.exp([p.logl_pi for p in points]),
        )

    def samples(self):
        return self.points_to_samples(self.dead)


class SequentialDiffusion(Integrator):
    beta_min = 0.00001
    beta_max = 1.0

    def run(self, n=1000, steps=10, schedule=np.linspace, **kwargs):
        target_eff = kwargs.get("efficiency", 0.5)

        betas = schedule(self.beta_min, self.beta_max, steps)
        self.dist = self.prior
        diffuser = self.model(self.prior)

        for beta_i in betas:
            live = self.sample(n * 2, self.dist, beta=beta_i)
            frame = self.points_to_samples(live)

            ess = len(frame.compress())
            while ess < n:
                live += self.sample(n, self.dist, beta=beta_i)
                frame = self.points_to_samples(live)
                ess = len(frame.compress())
                logger.info(f"Efficiency at: {ess/len(live)}, using previous diffusion")
            logger.info(f"Met ess criteria, training new diffusion")
            diffuser = self.model(self.prior)
            diffuser.train(
                np.asarray(self.points_to_samples(live).compress()),
                n_epochs=n,
                batch_size=n,
                lr=1e-3,
            )

            self.dist = diffuser
            self.dead += live
            self.update_stats(live, n)
            logger.info(f"{self.stats}")

        self.dead = self.sample(n * 4, self.dist)

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)
        self.stats.ndead = len(running_samples.compress())
        self.stats.logz = np.log(running_samples.get_weights().mean())

    def points_to_samples(self, points):
        if not points:
            return MCMCSamples(data=[], weights=[])
        else:
            logls = np.asarray([p.logl for p in points])
            logls -= logls.max()
            return MCMCSamples(data=[p.x for p in points], weights=np.exp(logls))
