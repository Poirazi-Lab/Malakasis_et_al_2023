#!/bin/bash

for iter in `seq 1 400`; do


neurons=$(shuf -n1 -e 50 100 200 250)
branches=$(shuf -n1 -e 5 10)
syns=$(shuf -n1 -e 1000 3000 3500)
syns_2=$(shuf -n1 -e 2 3)             
syns_3=$(shuf -n1 -e 1 2 4 5)
wmax=$(shuf -n1 -e 0.2 0.3) 
wmin=$(shuf -n1 -e 0 0.2)
alpha=$(shuf -n1 -e 10 50 100 500)
mid=$(shuf -n1 -e 0.4 0.5 0.6)
min_lr=$(shuf -n1 -e 0.001 0.00001)
turniters=$(shuf -n1 -e 20 30 40)
rewire=$(shuf -n1 -e 1 0.9 0.8)
turnthresh=$(shuf -n1 -e 0 0.05)
max_freq=$(shuf -n1 -e 20 30 40 60)
label_freq=$(shuf -n1 -e 35 40 50 60)
stop=$(shuf -n1 -e 0.5 0.8 0.9)

for run in `seq 1 10`; do

	LAPARAMS="-N ${neurons} -B ${branches} -G -S 1980${run} -o digit1=1 -o digit2=8 -s nikos_${iter}_${run}_digits_1_8_neurons_${neurons}_branches_${branches}_syns_${syns}_labsyns_${syns_2}_inhsyns_${syns_3}_Wmax_${wmax}_Wmin_${wmin}_slope_${alpha}_midpos_${mid}_minlr_${min_lr}_turnoveriters_${turniters}_rewireperc_${rewire}_turnoverthreshold_${turnthresh}_maxfreq_${max_freq}_labelfreq_${label_freq}_stopperc_${stop} -o inh_control_synapses=500 -o pattern_to_hidden_synapses=$syns -o label_to_hidden_synapses=$syns_2 -o lateral_inh_synapses=$syns_3 -o Wmin=$wm-o Wmax=$wmax -o slope=$alpha -o midpos=$mid -o maxlr=0.01 -o minlr=$min_lr -o iterations_to_turnover=$turniters -o rewired_percentile=$rewire -o turnover_threshold=$turnthresh -o active_neuron_spike_threshold=20 -o max_input_frequency=$max_freq -o label_input_frequency=$label_freq -o stop_threshold=$stop"
	echo $LAPARAMS
	qsub -v "LAPARAMS=$LAPARAMS" submit_lamodel.sh


done
done