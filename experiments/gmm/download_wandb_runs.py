import re
import os

from plot_train_alpha_study import get_wandb_runs


if __name__ == '__main__':
    alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    for with_buff in [False, True]:
        print(with_buff)
        fab_type = "buff" if with_buff else "no_buff"
        for alpha in alphas:
            runs = get_wandb_runs(alpha=alpha, with_buff=with_buff, n_runs=5)
            iter_n = 52076 if with_buff else 78125
            for i, run in enumerate(runs):
                for file in run.files():
                    if re.search(f'iter_{iter_n}/model.pt', str(file)):
                        name_without_seed = f"{fab_type}_alpha{alpha}"
                        file.download()
                        path = re.search(r"(results[^\s]*)", str(file)).group()
                        os.replace(path, f"./models_alpha/{name_without_seed}_seed{i}.pt")
                        print("saved" + path)
