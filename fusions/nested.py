import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from dataclasses import dataclass, field

import jax
import matplotlib.pyplot as plt
import numpy as np
from anesthetic import MCMCSamples, NestedSamples
from scipy.special import logsumexp
from tqdm import tqdm

os.makedirs("plots", exist_ok=True)


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


def plot_points(points):
    plt.scatter(*np.asarray([p.x for p in points]).T, c=[p.logl for p in points])
    # plt.show()


class Integrator(object):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -1e30)
        self.dead = []
        self.dists = []
        self.stats = Stats()

    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        x = np.asarray(dist.rvs(n))
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        # log_pi = prior(x) * (1-beta)
        points = [Point(x[i], logl[i], logl_birth[i]) for i in range(x.shape[0])]
        return points

    def stash(self, points, n):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        self.dead += live[n:]
        contour = live[n].logl
        live = live[:n]
        return live, contour


class NestedDiffusion(Integrator):
    def sample(self, n, dist, prior=lambda x: 0.0, logl_birth=0.0, beta=1.0):
        x = np.asarray(dist.rvs(n))
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        # log_pi = prior(x) * (1-beta)
        points = [Point(x[i], logl[i], logl_birth[i]) for i in range(x.shape[0])]
        return points

    def sample_constrained(self, n, dist, constraint, efficiency=0.5):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
            if len(batch_success) < n * efficiency:
                return False, len(success) / trials, batch_success
        return True, len(success) / trials, success

    def stash(self, points, n, terminate=False, prior=False):
        live = sorted(points, key=lambda lp: lp.logl, reverse=True)
        # if terminate:
        #     mean_logl = np.mean([p.logl for p in live])

        #     def average_live(x):
        #         x.logl = mean_logl
        #         return x

        #     live = [average_live(p) for p in live]
        # if not prior:
        #     contour = live[-n].logl
        #     for p in live[-n:]:
        #         p.logl_birth = contour
        #     for p in live[:-n][::-1]:
        #         p.logl_birth = contour
        #         contour = p.logl

        self.dead += live[n:]
        contour = live[n].logl
        # self.last_live = live[:-n]
        # if terminate:
        #     contour = 0
        # else:
        #     contour = live[-n].logl
        live = live[:n]
        return live, contour

    def run(self, n=1000, target_eff=0.1, steps=None, eps=-3):
        # self.stats.nlive = n
        live = self.sample(n * 2, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n, prior=True)
        self.dist = self.prior
        diffuser = Diffusion(self.prior)
        while step < steps:
            # if self.stats.logX<-3:
            #     self.stash(live, -len(live))
            #     logger.info("Precision reached, terminating")
            #     break
            # success, eff, points = self.sample_constrained(
            #     n, self.prior, contour, efficiency=target_eff
            # )
            success, eff, points = self.sample_constrained(
                n, self.dist, contour, efficiency=target_eff
            )
            if success:
                live, contour = self.stash(live + points, n)
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
                self.update_stats(live, n)
                logger.info(f"{self.stats}")
                step += 1

            if not (success):
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                # diffuser = CFM(self.prior)
                diffuser = Diffusion(self.prior)
                # with jax.disable_jit():
                prev_params = None
                if len(self.dists) > 1:
                    prev_params = self.dists[-1].state.params
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=n,
                    # batch_size=n // 2,
                    batch_size=n,
                    lr=1e-3,
                    params=prev_params,
                )
                live += points
                # live, contour = self.stash(live + points, n)
                MCMCSamples(diffuser.rvs(100)).plot_2d(np.arange(5))
                plt.savefig(f"plots/step_{step}.pdf")
                plt.close()
                self.dists.append(diffuser)
                # self.prior = diffuser
                self.dist = diffuser

            # step += 1
            logger.info(f"Step {step}/{steps} complete")

        # mean_logl = np.mean([p.logl for p in live])
        # average_live = lambda x: x.logl = mean_logl
        # live = [average_live(p) for p in live]
        self.stash(live, -len(live), terminate=True)
        print("done")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead + live)
        live_samples = self.points_to_samples(live)
        self.stats.ndead = len(self.dead)
        lZs = running_samples.logZ(100)
        self.stats.logX = -np.asarray(
            [(x.logl_birth > self.logzero) for x in self.dead]
        ).sum() / (self.stats.nlive)
        # self.stats.logX = live_samples.logZ() - running_samples.logZ()
        # self.stats.logX = logsumexp(live_samples.logX() * live_samples.logL)
        self.stats.logz = lZs.mean()
        self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        return NestedSamples(
            data=[p.x for p in points],
            logL=[p.logl for p in points],
            logL_birth=[p.logl_birth for p in points],
        )

    def samples(self):
        return self.points_to_samples(self.dead)


class SequentialDiffusion(NestedDiffusion):
    beta_min = 0.001
    beta_max = 1.0
    steps = 100
    schedule = np.linspace(beta_min, beta_max, steps)

    def sample_to_ess(self, points, ess):
        self.sample(n, self.dist)

    def run(self, n=1000, target_eff=0.1, steps=None, eps=-3):
        self.stats.nlive = n
        # live = self.sample(n * 2, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        self.dist = self.prior
        diffuser = Diffusion(self.prior)
        for beta_i in self.schedule:
            live = self.sample(n * 100, self.dist, beta=beta_i)
            frame = self.points_to_samples(live)
            ess = len(frame.compress("equal"))
            while ess < target_eff * n:
                live += self.sample(n, self.dist, beta=beta_i)
                frame = self.points_to_samples(live)
                ess = len(frame.compress("equal"))
                print(ess)
                # print(self.stats)
            diffuser.train(
                np.asarray(self.points_to_samples(live).compress("equal")),
                n_epochs=n,
                batch_size=n,
                lr=1e-3,
            )
            self.dist = diffuser
            self.dead += live

        self.dead = self.sample(10000, self.dist)

        # self.dead = self.sample(n*10, self.dist, beta=1.0)

        # while step < steps:
        #     while (
        #         len(self.points_to_samples(live).compress("equal"))
        #         < target_eff * n
        #     ):
        #         live += self.sample(n, self.dist, self.logzero)
        #     self.points_to_samples(live)
        #     if success:
        #         live, contour = self.stash(live + points, n)
        #         logger.info(f"Efficiency at: {eff}, using previous diffusion")
        #         self.update_stats(live, n)
        #         logger.info(f"{self.stats}")
        #         step += 1

        #     if not (success):
        #         logger.info(
        #             f"Efficiency dropped to: {eff}, training new diffusion"
        #         )
        #         # diffuser = CFM(self.prior)
        #         diffuser = Diffusion(self.prior)
        #         # with jax.disable_jit():
        #         prev_params = None
        #         if len(self.dists) > 1:
        #             prev_params = self.dists[-1].state.params
        #         diffuser.train(
        #             np.asarray([yi.x for yi in live + points]),
        #             n_epochs=n,
        #             # batch_size=n // 2,
        #             batch_size=n,
        #             lr=1e-3,
        #             params=prev_params,
        #         )
        #         live += points
        #         # live, contour = self.stash(live + points, n)
        #         MCMCSamples(diffuser.rvs(100)).plot_2d(np.arange(5))
        #         plt.savefig(f"plots/step_{step}.pdf")
        #         plt.close()
        #         self.dists.append(diffuser)
        #         # self.prior = diffuser
        #         self.dist = diffuser

        #     # step += 1
        #     logger.info(f"Step {step}/{steps} complete")

        # # mean_logl = np.mean([p.logl for p in live])
        # # average_live = lambda x: x.logl = mean_logl
        # # live = [average_live(p) for p in live]
        # self.stash(live, -len(live), terminate=True)
        # print("done")

    def points_to_samples(self, points):
        if not points:
            return MCMCSamples(data=[], weights=[])
        else:
            logls = np.asarray([p.logl for p in points])
            logls -= logls.max()
            return MCMCSamples(data=[p.x for p in points], weights=np.exp(logls))
