import logging

from fusions.cfm import CFM
from fusions.diffusion import Diffusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from anesthetic import MCMCSamples, NestedSamples
from scipy.special import logsumexp
from tqdm import tqdm

from fusions.model import Model


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


# def plot_points(points):
#     plt.scatter(
#         *np.asarray([p.x for p in points]).T, c=[p.logl for p in points]
#     )


class Integrator(ABC):
    def __init__(self, prior, likelihood, **kwargs) -> None:
        self.prior = prior
        self.likelihood = likelihood
        self.logzero = kwargs.get("logzero", -np.inf)
        self.dead = []
        self.dists = []
        self.stats = Stats()
        self.model = kwargs.get("model", Diffusion)

    def sample(self, n, dist, logl_birth=0.0, beta=1.0, calibrate=False):
        x = np.asarray(dist.rvs(n))
        # if calibrate:
        #     c = np.asarray(dist.predict_weight(x)).squeeze()
        #     # weights = c / (1-c)
        #     weights = 1 / c
        # else:
        #     weights = np.ones(x.shape[0])
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        log_pi = self.prior.logpdf(x)
        # logl += log_pi * (1.0 - beta)
        # frame = MCMCSamples(data=x, weights=weights)
        # frame["logl"] = logl
        # frame["logl_birth"] = logl_birth
        # frame["logl_pi"] = log_pi

        # frame = frame.compress()
        # points = [
        #     Point(
        #         frame.iloc[ii][np.arange(x.shape[-1])].to_numpy(),
        #         logl=frame.iloc[ii].logl,
        #         logl_birth=frame.iloc[ii].logl_birth,
        #     )
        #     for ii in range(len(frame))
        # ]
        points = [
            Point(x[i], logl[i], logl_birth[i], logl_pi=log_pi[i])
            for i in range(x.shape[0])
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


class NestedDiffusion(Integrator):
    def sample_constrained(self, n, dist, constraint, efficiency=0.5, **kwargs):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint, **kwargs)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
            if len(batch_success) < n * efficiency:
                return False, len(success) / trials, batch_success
        return True, len(success) / trials, success

    def run(self, n=1000, target_eff=0.1, steps=20):
        live = self.sample(n * 10, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        diffuser = self.model(self.prior)

        while step < steps:
            success, eff, points = self.sample_constrained(
                n,
                self.dist,
                contour,
                efficiency=target_eff,
                calibrate=isinstance(self.dist, Model),
            )
            step += 1
            live, contour = self.stash(live + points, n)

            if success:
                live, contour = self.stash(live + points, n)
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
            self.update_stats(live, n)
            logger.info(f"{self.stats}")
            step += 1
            if not (success):
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                # diffuser = Diffusion(self.prior)
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    # self.points_to_samples_importance(live + points)
                    # .compress()
                    # .to_numpy(),
                    n_epochs=len(live),
                    # n_epochs=len(live) * 2,
                    batch_size=n,
                    # batch_size=n // 2,
                    lr=1e-3,
                )
                live += points
                self.dists.append(diffuser)
                # self.prior = diffuser
                self.dist = diffuser
                # diffusion_samples = diffuser.rvs(100000)
                # diffuser.calibrate(diffusion_samples)
                # # diffuser.predict_weight(diffusion_samples)

            # diffusion_samples = diffuser.rvs(100000)
            # diffuser.calibrate(diffusion_samples)
            logger.info(f"Step {step}/{steps} complete")

        self.stash(live, -len(live))
        logger.info(f"Final stats: {self.stats}")

    def update_stats(self, live, n):
        running_samples = self.points_to_samples(self.dead)

        # running_samples = self.points_to_samples(self.dead + live)
        # live_samples = self.points_to_samples(live)
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

    def points_to_samples_importance(self, points, weights):
        return MCMCSamples(
            data=[p.x for p in points],
            weights=weights,
            # weights=np.exp([p.logl_pi for p in points]),
        )

    def write(self, filename="diffuser", dir="chains"):
        os.makedirs(dir, exist_ok=True)

    def samples(self):
        return self.points_to_samples(self.dead)


class SimpleNestedDiffusion(NestedDiffusion):
    def sample_constrained(self, n, dist, constraint, efficiency=0.5, **kwargs):
        success = []
        trials = 0
        while len(success) < n:
            batch_success = []
            pi = self.sample(n, dist, constraint, **kwargs)
            batch_success += [p for p in pi if p.logl > constraint]
            success += batch_success
            trials += n
        return True, len(success) / trials, success

    def run(self, n=1000, target_eff=0.1, steps=20):
        live = self.sample(n * 10, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n, drop=False)
        self.dist = self.prior
        self.update_stats(live, n)
        logger.info(f"{self.stats}")
        diffuser = self.model(self.prior)

        while step < steps:
            success, eff, points = self.sample_constrained(
                n,
                self.dist,
                contour,
                efficiency=target_eff,
                calibrate=isinstance(self.dist, Model),
            )
            step += 1
            live, contour = self.stash(live + points, n)
            self.update_stats(live, n)
            logger.info(f"{self.stats}")

            diffuser.train(
                np.asarray([yi.x for yi in live]),
                # .to_numpy(),
                n_epochs=len(live) * 10,
                # n_epochs=len(live) * 2,
                batch_size=n // 2,
                # batch_size=n // 2,
                lr=1e-3,
            )
            self.dists.append(diffuser)
            self.dist = diffuser
            # diffusion_samples = diffuser.rvs(100000)
            # diffuser.calibrate(diffusion_samples)
            logger.info(f"Step {step}/{steps} complete")

        self.stash(live, -len(live))
        logger.info(f"Final stats: {self.stats}")


class NestedSequentialDiffusion(NestedDiffusion):
    def sample(self, n, dist, logl_birth=0.0, beta=1.0):
        x = np.asarray(dist.rvs(n))
        logl = self.likelihood.logpdf(x) * beta
        self.stats.nlike += n
        logl_birth = np.ones_like(logl) * logl_birth
        # log_pi = self.prior.logpdf(x)
        # logl += log_pi * (1.0 - beta)
        points = [
            Point(x[i], logl[i], logl_birth[i])  # , logl_pi=log_pi[i])
            for i in range(x.shape[0])
        ]
        return points

    def run(self, n=1000, target_eff=0.1, steps=20):
        live = self.sample(n * 2, self.prior, self.logzero)
        step = 0
        logger.info("Done sampling prior")
        live, contour = self.stash(live, n)

        while step < steps:
            success, eff, points = self.sample_constrained(
                n, self.prior, contour, efficiency=target_eff
            )
            if success:
                live, contour = self.stash(live + points, n)
                logger.info(f"Efficiency at: {eff}, using previous diffusion")
                self.update_stats(live, n)
                logger.info(f"{self.stats}")
                step += 1

            if not (success):
                logger.info(f"Efficiency dropped to: {eff}, training new diffusion")
                diffuser = CFM(self.prior)
                diffuser.train(
                    np.asarray([yi.x for yi in live + points]),
                    n_epochs=len(live) * 2,
                    # batch_size=n // 2,
                    batch_size=n // 2,
                    lr=1e-3,
                )
                live += points
                self.dists.append(diffuser)
                self.prior = diffuser

            logger.info(f"Step {step}/{steps} complete")

        self.stash(live, -len(live))
        logger.info(f"Final stats: {self.stats}")


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
            # self.points_to_samples(live).plot_2d(np.arange(5))
            # plt.savefig("plots/step_{}.pdf".format(beta_i))
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

        # # live_samples = self.points_to_samples(live)
        self.stats.ndead = len(running_samples.compress())
        # lZs = running_samples.logZ(100)
        # self.stats.logX = -np.asarray(
        #     [(x.logl_birth > self.logzero) for x in self.dead]
        # ).sum() / (self.stats.nlive)
        # # self.stats.logX = live_samples.logZ() - running_samples.logZ()
        # # self.stats.logX = logsumexp(live_samples.logX() * live_samples.logL)
        self.stats.logz = np.log(running_samples.get_weights().mean())
        # self.stats.logz_err = lZs.std()

    def points_to_samples(self, points):
        if not points:
            return MCMCSamples(data=[], weights=[])
        else:
            logls = np.asarray([p.logl for p in points])
            logls -= logls.max()
            return MCMCSamples(data=[p.x for p in points], weights=np.exp(logls))
