import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pickle import dump, load

import matplotlib.pyplot as plt
import numpy as np
from jax import random

# from anesthetic.read.hdf import read_hdf, write_hdf
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from tqdm import tqdm

import anesthetic.termination as term
from anesthetic import MCMCSamples, NestedSamples, make_2d_axes, read_chains
from anesthetic.utils import compress_weights, neff
from fusions.model import Model
from fusions.utils import ellipse, unit_hyperball, unit_hypercube


@dataclass
class Point:
    x: np.ndarray
    latent_x: np.ndarray
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
    """Settings for the integrator.

    Args:
        n (int, optional): Number of samples to draw. Defaults to 500.
        target_eff (float, optional): Target efficiency. Defaults to 0.1.
        steps (int, optional): Number of steps to take. Defaults to 20.
        prior_boost (int, optional): Number of samples to draw from the prior. Defaults to 5.
        eps (float, optional): Tolerance for the termination criterion. Defaults to 1e-3.
        batch_size (float, optional): Batch size for training the diffusion. Defaults to 0.25.
        epoch_factor (int, optional): Factor to multiply the number of epochs by. Defaults to 1.
        restart (bool, optional): Whether to restart the training. Defaults to False.
        noise (float, optional): Noise to add to the training. Defaults to 1e-3.
        efficiency (float, optional): Efficiency. Defaults to 1 / np.e.
        logzero (float, optional): Value to use for log zero. Defaults to -1e30.
    """

    n: int = 500
    target_eff: float = 0.1
    steps: int = 20
    prior_boost: int = 5
    eps: float = 1e-3
    batch_size: float = 0.25
    epoch_factor: int = 1
    restart: bool = False
    noise: float = 1e-3
    resume: bool = False
    dirname: str = "fusions_samples"
    lr: float = 1e-3
    # efficiency: float = 1 / np.e
    # logzero: float = -1e30

    def __repr__(self):
        return (
            f"Settings(\n"
            f"  n: {self.n},\n"
            f"  target_eff: {self.target_eff},\n"
            f"  steps: {self.steps},\n"
            f"  prior_boost: {self.prior_boost},\n"
            f"  eps: {self.eps},\n"
            f"  batch_size: {self.batch_size},\n"
            f"  epoch_factor: {self.epoch_factor},\n"
            f"  restart: {self.restart},\n"
            f"  noise: {self.noise},\n"
            f"  resume: {self.resume},\n"
            # f"  efficiency: {self.efficiency},\n"
            f")"
        )


@dataclass
class Trace:
    diff: Point = field(default_factory=dict)
    live: Point = field(default_factory=dict)
    prior: Point = field(default_factory=dict)
    accepted_live: Point = field(default_factory=dict)
    iteration: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=dict)
    efficiency: list[float] = field(default_factory=list)


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
        # latent = kwargs.get("latent", unit_hyperball)
        # latent = multivariate_normal(mean=np.zeros(prior.dim), cov=np.eye(prior.dim))
        # self.latent = latent(prior.dim, scale=1.0)
        self.dim = prior.dim
        self.rng = kwargs.get("rng", random.PRNGKey(0))
        self.trace = Trace()
        # self.prior = multivariate_normal(
        #     np.zeros(prior.dim), np.eye(prior.dim)
        # )
        self.latent = multivariate_normal(np.zeros(prior.dim), np.eye(prior.dim))

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        if isinstance(dist, Model):
            # n = int(n * 10)
            # n=int(n*5)
            x, j = dist.rvs(n, jac=True, solution="exact")
            logging.log(logging.INFO, f"Sampling {n} points")
            latent_x = dist.prior.rvs(n)
            self.rng, key = random.split(self.rng)
            # noise = random.normal(key, (n, self.dim)) * 1
            # x = dist.predict(latent_x , jac=False)
            x = np.asarray(x)
            latent_x = np.asarray(latent_x)
            w = np.ones(n)
            log_pi = self.prior.logpdf(x)
            # w = dist.predict_weight(x).flatten()
            # print(w.mean(),w.std())
            w = np.exp(2 * log_pi - j)
            # w=1/j
        else:
            x = np.asarray(dist.rvs(n))
            w = np.ones(n)
            latent_x = x
        log_pi = self.prior.logpdf(x)

        idx = compress_weights(w.flatten(), ncompress="equal")
        idx = np.asarray(idx, dtype=bool)

        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += idx.sum()
        logl_birth = np.ones_like(logl) * logl_birth
        points = [
            Point(
                x[idx][i],
                latent_x[idx][i],
                logl[idx][i],
                logl_birth[idx][i],
                logl_pi=log_pi[idx][i],
            )
            for i in range(idx.sum())
        ]
        return points

    def stash(self, points, n, drop=False):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        if not drop:
            self.dead += live[n:]
        contour = live[n].logl
        r = np.linalg.norm(live[n].latent_x)
        live = live[:n]
        return live, contour, r

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
        os.makedirs(self.settings.dirname, exist_ok=True)
        self.points_to_samples(self.dead).to_csv(
            os.path.join(self.settings.dirname, filename) + ".csv"
        )

    def read(self, filename):
        self.dead = read_chains(filename)

    def write_trace(self, filename):
        os.makedirs(self.settings.dirname, exist_ok=True)
        dump(
            self.trace,
            open(os.path.join(self.settings.dirname, filename) + ".pkl", "wb"),
        )


class NestedDiffusion(Integrator):
    def sample_constrained(self, n, dist, constraint, efficiency=0.1, **kwargs):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint, **kwargs)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += len(pi)
        eff = len(success) / trials
        return eff > efficiency, eff, success

    def train_diffuser(self, dist, points, prior_samples=None):
        dist.train(
            np.asarray([yi.x for yi in points]),
            n_epochs=int(len(points) * self.settings.epoch_factor),
            # batch_size=int(len(points) * self.settings.batch_size),
            batch_size=self.settings.batch_size,
            lr=self.settings.lr,
            restart=self.settings.restart,
            noise=self.settings.noise,
            prior_samples=prior_samples,
        )
        return dist

    def run(self):
        print(self.settings)
        # if settings.resume:
        #     try:
        #         points = self.read()
        n = self.settings.n
        live = self.sample(n * self.settings.prior_boost, self.prior, self.logzero)

        step = 0
        logger.info("Done sampling prior")
        live, contour, r = self.stash(live, n // 2, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        self.dist = self.model(self.latent, noise=1e-3)
        dists = []
        # self.dist = self.train_diffuser(diffuser, live)
        self.dist = self.model(self.prior, noise=self.settings.noise)

        # diffuser.rng, _ = random.split(diffuser.rng)
        r_true = 1.0
        while not self.points_to_samples(live + self.dead).terminated(
            "logZ", self.settings.eps
        ):
            # dist = self.model(ellipse(live))
            # print(r_true)

            # class flow(object):
            #     def __init__(self, dists, prior):
            #         self.dists = dists
            #         self.prior = prior

            #     def logpdf(self, x):
            #         return self.prior.logpdf(x)

            #     def rvs(self, n):
            #         x = self.prior.rvs(n)
            #         for d in self.dists:
            #             x = d.predict(x)
            #         return x

            #     def __call__(self, x):
            #         self
            #         for d in self.dists:
            #             x = d.predict(x)
            #         return x

            # f = flow(dists, self.prior)
            # dist = self.model(f)
            # x_prior = self.prior.rvs(n*10)
            # prior_samples = f.rvs(len(live) // 2)
            # dist = self.model(unit_hyperball(self.dim, scale = r_true))
            # r_true/=2
            xi = np.random.choice(len(live), int(1.0 * len(live)), replace=False)
            # xi, ci = np.random.choice(len(live), (2, len(live) // 2), replace=False)
            # dist = self.train_diffuser(dist, np.asarray(live)[xi], prior_samples)
            # x = dist.predict(prior_samples)
            # x = self.dist.rvs(len(live) // 2)
            # # self.trace.diff[step] = np.asarray(x)
            # self.dist.calibrate(
            #     np.asarray([yi.x for yi in np.asarray(live)[ci]]),
            #     np.asarray(x),
            #     n_epochs=len(live) * 5,
            #     batch_size=n // 2,
            #     restart=True,
            # )
            # self.dist = self.train_diffuser(self.dist, np.asarray(live)[xi])
            self.dist = self.train_diffuser(self.dist, live)

            # x = self.dist.rvs(len(live) // 2)
            # self.trace.diff[step] = np.asarray(x)
            # self.dist.calibrate(
            #     np.asarray([yi.x for yi in np.asarray(live)[ci]]),
            #     np.asarray(x),
            #     n_epochs=len(live) * 5,
            #     batch_size=n // 2,
            #     restart=True,
            # )
            self.trace.losses[step] = self.dist.trace.losses

            # live, contour = self.stash(live, n//2, drop=False)
            # self.dist = self.train_diffuser(self.dist, live)
            success, eff, points = self.sample_constrained(
                n // 2,
                self.dist,
                contour,
                efficiency=self.settings.target_eff,
            )
            logger.info(f"Efficiency at: {eff}, using previous diffusion")

            self.trace.live[step] = live
            self.trace.accepted_live[step] = points
            self.trace.prior[step] = self.dist.prior.rvs(n)
            live = live + points
            live, contour, r = self.stash(live, n // 2, drop=False)
            # dists.append(dist)

            # self.dist = self.train_diffuser(self.dist, live)
            # x = self.dist.rvs(len(live))
            # self.dist.calibrate(np.asarray([yi.x for yi in live]), np.asarray(x), n_epochs=len(live) * 5, batch_size=n//2, restart =True)
            # self.trace.losses[step] = self.dist.trace.losses

            # live, contour = self.stash(live, n // 2)
            # x = self.dist.rvs(len(live))
            # self.trace.diff[step] = x
            # t = np.asarray([yi.x for yi in live])
            # self.dist.calibrate(np.asarray([yi.x for yi in live]), np.asarray(x), n_epochs=len(live) * 5, batch_size=n, restart =True)

            # if success:
            #     logger.info(f"Efficiency at: {eff}, using previous diffusion")
            # if not success:
            #     logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
            #     diffuser = self.train_diffuser(diffuser, live)
            #     self.trace.losses[step] = diffuser.trace.losses
            #     self.dists.append(diffuser)
            #     self.dist = diffuser

            self.update_stats(live, n)
            logger.info(f"{self.stats}")
            step += 1
            self.trace.iteration.append(step)
            self.trace.efficiency.append(eff)
            self.write_trace("trace")

            logger.info(f"Step {step} complete")

        self.stash(live, -len(live))
        self.write_trace("trace")
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead + live)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        # self.stats.logX = running_samples.critical_ratio()
        running_samples.terminated()
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
