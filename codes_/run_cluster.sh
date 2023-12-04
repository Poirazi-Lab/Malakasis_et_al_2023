#!/bin/bash

for run in {1..20}; do
for digit1 in {0..9}; do
for digit2 in {0..9}; do

	if ((${digit1} < ${digit2})); then
	LAPARAMS="-N 100 -B 10 -G -S 2020${run} -o digit1=${digit1} -o digit2=${digit2} -s nikos_${run}_digits_${digit1}_${digit2} -o inh_control_synapses=500 -o pattern_to_hidden_synapses=1750 -o label_to_hidden_synapses=2 -o lateral_inh_synapses=4 -o Wmin=0.1 -o Wmax=0.2 -o slope=10 -o midpos=0.5 -o maxlr=0.001 -o minlr=0.0001 -o iterations_to_turnover=20 -o rewired_percentile=1 -o turnover_threshold=0 -o active_neuron_spike_threshold=20 -o max_input_frequency=25 -o label_input_frequency=40 -o stop_threshold=0.5 -o LoadNet=0 -o nlTypePYR=0"
	echo $LAPARAMS
	qsub -v "LAPARAMS=$LAPARAMS" submit_lamodel.sh
	fi

done
done
done
