import matplotlib.pyplot as plt
import numpy as np
import wandb
api = wandb.Api()

alphas = [1.0, 1.5, 2.0, 3.0]  # 0.25, 0.5,


def get_wandb_runs(alpha, with_buff, n_runs = 3):
    filters = {"$and": [{"tags": "iclr_rebuttal"},
                        {"tags": "exp2"},
                        {"tags": "thurs"},
                        {"config.training": {"$regex": f"'use_buffer': {with_buff},"}}
                        ]}
    if alpha % 1 == 0:
        filters['$and'].append(
            {"$or":
                                 [{"config.fab": {"$regex": f"'alpha': {alpha},"}},
                                  {"config.fab": {"$regex": f"'alpha': {int(alpha),}"}}
                                  ]}
        )
    else:
        filters['$and'].append({"config.fab": {"$regex": f"'alpha': {alpha},"}})
    if not with_buff:
        filters['$and'].append({"tags": "bug_fix"})
    runs = api.runs(path='flow-ais-bootstrap/fab',
                    filters=filters)
    key = 'test_set_mean_log_prob_p_target' if with_buff else 'test_set_mean_log_prob'
    runs_list = []
    i = 0
    while not len(runs_list) == n_runs:
        if i >= (len(runs)):
            print(f"not enough seeds at alpha={alpha}:")
            print(f"only {len(runs_list)} seeds \n")
            break
        run = runs[i]
        history = run.history(keys=[key])
        if "finished" not in str(run) or key not in history.keys():
            i += 1
            continue
        runs_list.append(run)
        i += 1
    return runs_list


def get_runs(alpha, with_buff: bool, n_runs=3):
    runs = get_wandb_runs(alpha, with_buff, n_runs)
    n_steps = []
    log_probs = []
    for run in runs:
        key = 'test_set_mean_log_prob_p_target' if with_buff else 'test_set_mean_log_prob'
        history = run.history(keys=[key])
        n_steps.append(list(np.array(history['_step'])))
        log_probs.append(list(np.array(history[key], dtype=float)))
    return np.array(n_steps), np.array(log_probs)


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 2, sharey=True)
    for i, use_buff in enumerate([False, True]):
        axs[i].set_title("w buffer" if use_buff else "w/o buffer")
        for j, alpha in enumerate(alphas):
            print(f"plotted {alpha}")
            n_steps, log_probs = get_runs(alpha, use_buff)
            means = np.nanmean(log_probs, axis=0)
            std = np.nanstd(log_probs, axis=0)
            axs[i].plot(n_steps[0][np.isfinite(means)], means[np.isfinite(means)],
                        "-o", label=fr"$\alpha={alpha}$")
            # axs[i].set_title(fr"alpha={alpha}$")
            # axs[i].fill_between(n_steps[0][np.isfinite(means)],
            #                     (means - std)[np.isfinite(means)],
            #                     (means + std)[np.isfinite(means)],
            #                     alpha=0.1
            #                     )
            axs[i].set_ylim(-15, -5)
    # plt.yscale("symlog")
    axs[0].legend()
    axs[0].set_ylabel("log likelihood")
    axs[0].set_xlabel("training iteration")
    axs[1].set_xlabel("training iteration")
    # axs[2].set_ylabel("log likelihood")
    # axs[2].set_xlabel("training steps")
    # axs[3].set_xlabel("training steps")
    plt.tight_layout()
    plt.show()
