import pandas as pd
from experiments.gmm.evaluation import FILENAME_EVAL_INFO
from experiments.gmm.evaluation_expectation_quadratic_func import FILENAME_EXPECTATION_INFO



if __name__ == '__main__':
    df_eval_info = pd.read_csv(open(FILENAME_EVAL_INFO, "r"))
    df_expectation_info = pd.read_csv(open(FILENAME_EXPECTATION_INFO, "r"))


    keys1 = ["eval_ess_flow", "test_set_mean_log_prob", 'kl_forward']
    keys2 = ["bias", "bias_unweighted"]
    columns = ["target_kld", "flow_nis", "flow_kld", "rsb", "snf", "fab_no_buffer", "fab_buffer"]
    column_names = ["Flow w/ ML",
                    "Flow w/ $D_{\\alpha=2}$",
                    "Flow w/ KLD",
                    "RSB w/ KLD",
                    "SNF w/ KLD",
                    "\emph{FAB w/o buffer}",
                    "\emph{FAB w/ buffer}"]
    means_eval = df_eval_info.groupby("model_name").mean()[keys1]
    stds_eval = df_eval_info.groupby("model_name").sem(ddof=0)[keys1]
    means_exp = df_expectation_info.groupby("model_name").mean()[keys2]
    stds_exp = df_expectation_info.groupby("model_name").sem(ddof=0)[keys2]

    table_values_string = ""

    for i, column in enumerate(columns):
        mean_eval, std_eval, mean_exp, std_exp = means_eval.loc[column], stds_eval.loc[column], \
                                                 means_exp.loc[column], stds_exp.loc[column]
        if column == "target_kld":
            table_values_string += f"{column_names[i]} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[0]]*100:.1f}{'}'}," \
                                                r"\bm" + f"{'{'}{std_eval[keys1[0]]*100:.1f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[1]]:.2f}{'}'}," \
                                                r"\bm" + f"{'{'}{std_eval[keys1[1]]:.2f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[2]]:.2f}{'}'}," \
                                                r"\bm" + f"{'{'},{std_eval[keys1[2]]:.2f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_exp[keys2[0]]*100:.1f}{'}'}," \
                                                r"\bm" + f"{'{'}{std_exp[keys2[0]]*100:.1f}{'}'} & " \
                                    r"\bm" + f"{'{'}{mean_exp[keys2[1]]*100:.1f}{'}'}," \
                                                r"\bm" + f"{'{'}{std_exp[keys2[1]]*100:.1f}{'}'} \\\\ \n"
            table_values_string += "\hdashline \n "
        elif column == "fab_buffer":
            table_values_string += f"{column_names[i]} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[0]] * 100:.1f}{'}'}," \
                                   r"\bm" + f"{'{'}{std_eval[keys1[0]] * 100:.1f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[1]]:.2f}{'}'}," \
                                   r"\bm" + f"{'{'}{std_eval[keys1[1]]:.2f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_eval[keys1[2]]:.2f}{'}'}," \
                                   r"\bm" + f"{'{'}{std_eval[keys1[2]]:.2f}{'}'} & " \
                                   r"\bm" + f"{'{'}{mean_exp[keys2[0]] * 100:.1f}{'}'}," \
                                   r"\bm" + f"{'{'}{std_exp[keys2[0]] * 100:.1f}{'}'} & " \
                                   f"{mean_exp[keys2[1]] * 100:.1f}," \
                                   f"{std_exp[keys2[1]] * 100:.1f}\\\\ \n"
        else:
            table_values_string += f"{column_names[i]} & " \
                                   f"{mean_eval[keys1[0]]*100:.1f},{std_eval[keys1[0]]*100:.1f} & " \
                                   f"{mean_eval[keys1[1]]:.2f},{std_eval[keys1[1]]:.2f} & " \
                                   f"{mean_eval[keys1[2]]:.2f},{std_eval[keys1[2]]:.2f} & " \
                                   f"{mean_exp[keys2[0]]*100:.1f},{std_exp[keys2[0]]*100:.1f} & " \
                                    f"{mean_exp[keys2[1]]*100:.1f},{std_exp[keys2[1]]*100:.1f} \\\\ \n"
        #table_values_string += "" if i == len(columns) - 1 else ""
        if column != "snf":
            table_values_string = table_values_string.replace("nan", "\\text{NaN}")
        else:
            table_values_string = table_values_string.replace("nan", "\\text{N/A}")
    print(table_values_string)
