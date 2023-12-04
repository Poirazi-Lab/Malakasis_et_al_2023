for run in {1..200}; do

	LAPARAMS="-N 100 -B 10 -G -S 2020${run} -o digit1=3 -o digit2=8 -s nikos_${run}_digits_3_8 -o inh_control_synapses=500 -o pattern_to_hidden_synapses=1750 -o label_to_hidden_synapses=2 -o lateral_inh_synapses=4 -o Wmin=0.1 -o Wmax=0.2 -o slope=10 -o midpos=0.5 -o maxlr=0.01 -o minlr=0.001 -o iterations_to_turnover=20 -o rewired_percentile=1 -o turnover_threshold=0 -o active_neuron_spike_threshold=20 -o max_input_frequency=25 -o label_input_frequency=40 -o stop_threshold=0.5 -o LoadNet=0 -o nlTypePYR=0"
	echo $LAPARAMS
	qsub -v "LAPARAMS=$LAPARAMS" submit_lamodel.sh

done

