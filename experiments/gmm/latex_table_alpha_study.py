import pandas as pd
from experiments.gmm.evaluation import FILENAME_EVA_ALPHA_INFO
from experiments.gmm.evaluation_expectation_quadratic_func import FILENAME_EXPECTATION_ALPHA_INFO



if __name__ == '__main__':
    df_eval_info = pd.read_csv(open(FILENAME_EVA_ALPHA_INFO, "r"))
    df_expectation_info = pd.read_csv(open(FILENAME_EXPECTATION_ALPHA_INFO, "r"))


    keys1 = ["eval_ess_flow", "test_set_mean_log_prob", 'kl_forward']
    keys2 = ["bias", "bias_unweighted"]
    alpha_values = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    means_eval = df_eval_info.groupby("model_name").mean()[keys1]
    stds_eval = df_eval_info.groupby("model_name").sem(ddof=0)[keys1]
    means_exp = df_expectation_info.groupby("model_name").mean()[keys2]
    stds_exp = df_expectation_info.groupby("model_name").sem(ddof=0)[keys2]

    table_values_string = ""

    i = 0
    for fab_type in ["buff"]:  # , "buff"]: "no_buff"
        for alpha in alpha_values:
            name_without_seed = f"{fab_type}_alpha{alpha}"
            name = name_without_seed
            mean_eval, std_eval, mean_exp, std_exp = means_eval.loc[name], stds_eval.loc[name], \
                                                     means_exp.loc[name], stds_exp.loc[name]
            column_name = alpha # fr"$\alpha={alpha}$ {'w/o buffer' if fab_type == 'no_buff' else 'w/ buffer'}"
            table_values_string += f"{column_name} & " \
                                   f"{mean_eval[keys1[0]]*100:.1f},{std_eval[keys1[0]]*100:.1f} & " \
                                   f"{mean_eval[keys1[1]]:.2f},{std_eval[keys1[1]]:.2f} & " \
                                   f"{mean_eval[keys1[2]]:.2f},{std_eval[keys1[2]]:.2f} & " \
                                   f"{mean_exp[keys2[0]]*100:.1f},{std_exp[keys2[0]]*100:.1f} & " \
                                    f"{mean_exp[keys2[1]]*100:.1f},{std_exp[keys2[1]]*100:.1f} \\\\ \n"
            table_values_string = table_values_string.replace("nan", "\\text{N/A}")
            i += 1
    print(table_values_string)
