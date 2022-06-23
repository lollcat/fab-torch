# Variables
ManyWellCMD=python examples/many_well.py -m training.seed=0 # ,1,2
BASE_DIR=/home/laurence/work/code/FAB-TORCH/results

mw_fab_buffer: # run fab with prioritised fab_buffer
	$(ManyWellCMD) training.use_buffer=True training.prioritised_buffer=True \
		training.checkpoint_load_dir=$(BASE_DIR)/fab_buffer \
		evaluation.save_path=$(BASE_DIR)/fab_buffer

mw_fab_no_buffer:
	$(ManyWellCMD) fab.loss_type=p2_over_q_alpha_2_div \
		training.checkpoint_load_dir=$(BASE_DIR)/fab_no_buffer \
		evaluation.save_path=$(BASE_DIR)/fab_no_buffer

mw_flow_kld:
	$(ManyWellCMD) fab.loss_type=flow_reverse_kl \
		training.checkpoint_load_dir=$(BASE_DIR)/flow_kld \
		evaluation.save_path=$(BASE_DIR)/flow_kld

mw_flow_nis:
	$(ManyWellCMD) fab.loss_type=flow_alpha_2_div_nis \
		training.checkpoint_load_dir=$(BASE_DIR)/flow_nis \
		evaluation.save_path=$(BASE_DIR)/flow_nis