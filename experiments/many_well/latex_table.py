import pandas as pd
from experiments.many_well.evaluation import FILENAME_EVAL_INFO



if __name__ == '__main__':
    df_eval_info = pd.read_csv(open(FILENAME_EVAL_INFO, "r"))
    keys1 = ['eval_ess_flow', 'test_set_exact_mean_log_prob', 'test_set_modes_mean_log_prob',
             "forward_kl", 'MSE_log_Z_estimate']
    columns = ["target_kld"]
    column_names = ["Flow w/ ML"]
    # columns = ["flow_nis", "flow_kld", "snf", "fab_no_buffer", "fab_buffer"]
    # column_names = ["Flow w/ $D_{\\alpha=2}$", "Flow w/ KLD",
    #                 "SNF w/ KLD", "\emph{FAB w/o buffer}", "\emph{FAB w/ buffer}"]
    # keys1 = ['eval_ess_flow', 'test_set_exact_mean_log_prob', 'test_set_modes_mean_log_prob',
    #          "forward_kl", 'MSE_log_Z_estimate', ]
    # columns = ["target_kld", "flow_nis", "flow_kld", "snf", "fab_no_buffer", "fab_buffer"]
    # column_names = ["Flow w/ ML", "Flow w/ $D_{\\alpha=2}$", "Flow w/ KLD",
    #                 "SNF w/ KLD", "\emph{FAB w/o buffer}", "\emph{FAB w/ buffer}"]
    means_eval = df_eval_info.groupby("model_name").mean()[keys1]
    stds_eval = df_eval_info.groupby("model_name").sem(ddof=0)[keys1]

    table_values_string = ""

    for i, column in enumerate(columns):
        mean_eval, std_eval = means_eval.loc[column], stds_eval.loc[column]
        if column == "snf":
            table_values_string += f"{column_names[i]} & " \
                                   f"{mean_eval[keys1[0]]*100:.1f},{std_eval[keys1[0]]*100:.1f} & " \
                                   "text{N/A},text{N/A} & " \
                                   "text{N/A},text{N/A} & " \
                                   f"{mean_eval[keys1[3]]:.1f},{std_eval[keys1[3]]:.1f} & " \
                                   f"{mean_eval[keys1[4]]*100:.1f},{std_eval[keys1[4]]*100:.1f} \\\\ \n"
        else:
            table_values_string += f"{column_names[i]} & " \
                                   f"{mean_eval[keys1[0]]*100:.1f},{std_eval[keys1[0]]*100:.1f} & " \
                                   f"{mean_eval[keys1[1]]:.1f},{std_eval[keys1[1]]:.2f} & " \
                                    f"{mean_eval[keys1[2]]:.1f},{std_eval[keys1[2]]:.1f} & " \
                                   f"{mean_eval[keys1[3]]:.1f},{std_eval[keys1[3]]:.1f} & " \
                                   f"{mean_eval[keys1[4]]*100:.1f},{std_eval[keys1[4]]*100:.1f} \\\\ \n"
        if column != "snf":
            table_values_string = table_values_string.replace("nan", "\\text{NaN}")
        else:
            table_values_string = table_values_string.replace("nan", "\\text{N/A}")
    print(table_values_string)
