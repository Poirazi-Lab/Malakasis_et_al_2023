#!/bin/bash

for run in {1..10}; do
for neurons in 50 100 200 250 500; do
for branches in 5 10 20; do
for syns in 1000 2000 3000 3500 4000 5000; do
for syns_2 in {1..3}; do                  
for syns_3 in {1..6}; do
for wmax in 0.2 0.3; do
for wmin in 0 0.1 0.2; do
for alpha in 10 50 100 250 500 1000; do
for mid in 0.3 0.4 0.5 0.6 0.7; do
for min_lr in 0.001 0.0001 0.00001; do
for turniters in 5 10 20 30 40; do
for rewire in 1 0.9 0.8 0.7; do
for turnthresh in 0 0.05 0.1; do
for max_freq in 20 25 30 35 40 50 60; do
for label_freq in 20 25 30 35 40 50 60; do
for stop in 0.5 0.6 0.7 0.8 0.9; do


	LAPARAMS="-N $neurons -B $branches -G -S 1980${run} -o digit1=3 -o digit2=7 -s nikos_${run}_digits_3_7_neurons_${neurons}_branches_${branches}_syns_${syns}_labsyns_${syns_2}_inhsyns_${syns_3}_Wmax_${wmax}_Wmin_${wmin}_slope_${alpha}_midpos_${mid}_minlr_${min_lr}_turnoveriters_${turniters}_rewireperc_${rewire}_turnoverthreshold_${turnthresh}_maxfreq_${max_freq}_labelfreq_${label_freq}_stopperc_${stop} -o inh_control_synapses=500 -o pattern_to_hidden_synapses=$syns -o label_to_hidden_synapses=$syns_2 -o lateral_inh_synapses=$syns_3 -o Wmin=$wmin -o Wmax=$wmax -o slope=$alpha -o midpos=$mid -o maxlr=0.01 -o minlr=$min_lr -o iterations_to_turnover=$turniters -o rewired_percentile=$rewire -o turnover_threshold=$turnthresh -o active_neuron_spike_threshold=20 -o max_input_frequency=$max_freq -o label_input_frequency=$label_freq -o stop_threshold=$stop"
	echo $LAPARAMS
	qsub -v "LAPARAMS=$LAPARAMS" submit_lamodel.sh


done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

