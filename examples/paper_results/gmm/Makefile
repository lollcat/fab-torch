# Variables
GMMCMD=python examples/gmm.py -m training.seed=0,1,2 # by default seeds are run in parellel
RESULTS_DIR=/home/laurence/work/code/FAB-TORCH/results
TIMELIMIT=None
N_FLOW_FORWARD_PASS = 10_000_000_000


gmm_fab_buffer: # run fab with prioritised fab_buffer
	$(GMMCMD) training.use_buffer=True training.prioritised_buffer=True \
		training.checkpoint_load_dir=$(RESULTS_DIR)/fab_buffer/seed$$\{training.seed\}  \
		evaluation.save_path=$(RESULTS_DIR)/fab_buffer/seed$$\{training.seed\} \
		training.tlimit=$(TIMELIMIT) training.n_flow_forward_pass=$(N_FLOW_FORWARD_PASS)

gmm_fab_no_buffer:
	$(GMMCMD) fab.loss_type=p2_over_q_alpha_2_div \
		training.checkpoint_load_dir=$(RESULTS_DIR)/fab_no_buffer/seed$$\{training.seed\}  \
		evaluation.save_path=$(RESULTS_DIR)/fab_no_buffer/seed$$\{training.seed\} \
		training.tlimit=$(TIMELIMIT) training.n_flow_forward_pass=$(N_FLOW_FORWARD_PASS)

gmm_flow_kld:
	$(GMMCMD) fab.loss_type=flow_reverse_kl \
		training.checkpoint_load_dir=$(RESULTS_DIR)/flow_kld/seed$$\{training.seed\}  \
		evaluation.save_path=$(RESULTS_DIR)/flow_kld/seed$$\{training.seed\} \
		training.tlimit=$(TIMELIMIT) training.n_flow_forward_pass=$(N_FLOW_FORWARD_PASS)

gmm_flow_nis:
	$(GMMCMD) fab.loss_type=flow_alpha_2_div_nis \
		training.checkpoint_load_dir=$(RESULTS_DIR)/flow_nis/seed$$\{training.seed\}\
		evaluation.save_path=$(RESULTS_DIR)/flow_nis/seed$$\{training.seed\} \
		training.tlimit=$(TIMELIMIT) training.n_flow_forward_pass=$(N_FLOW_FORWARD_PASS)


gmm_snf:
	$(GMMCMD) flow.use_snf=True \
		training.checkpoint_load_dir=$(RESULTS_DIR)/snf/seed$$\{training.seed\}\
		evaluation.save_path=$(RESULTS_DIR)/snf/seed$$\{training.seed\} \
		training.tlimit=$(TIMELIMIT) training.n_flow_forward_pass=$(N_FLOW_FORWARD_PASS)