for turn in 0 1 2; do
for syns in 500 1000 2000; do
for stab in 0 1 2; do
for over in 0 2 4; do
for run in {1..20}; do


	LAPARAMS="-T ./Shape_Datasets/Shapes_train_${stab}${over}${run} -t ./Shape_Datasets/Shapes_train_labels_${stab}${over}${run} -E ./Shape_Datasets/Shapes_test_${stab}${over}${run} -e ./Shape_Datasets/Shapes_test_labels_${stab}${over}${run} -N 100 -B 10 -G -S 2020${run} -o digit1=1 -o digit2=2 -s nikos_shapes_${stab}_${over}_${run}_${turn}_${syns} -o inh_control_synapses=500 -o pattern_to_hidden_synapses=${syns} -o label_to_hidden_synapses=2 -o lateral_inh_synapses=4 -o Wmin=0.1 -o Wmax=0.2 -o slope=10 -o midpos=0.5 -o maxlr=0.01 -o minlr=0.001 -o iterations_to_turnover=10 -o rewired_percentile=1 -o turnover_threshold=0 -o active_neuron_spike_threshold=20 -o max_input_frequency=30 -o label_input_frequency=35 -o stop_threshold=0.5 -o LoadNet=0 -o nlTypePYR=0 -o Turnover_Type=${turn}"
	echo $LAPARAMS
	qsub -v "LAPARAMS=$LAPARAMS" submit_lamodel_shapes.sh

done
done
done
done
done