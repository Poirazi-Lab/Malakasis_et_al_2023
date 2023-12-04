// Version: $Id: lamodel.cpp 172 2014-02-12 10:06:07Z gk $
/* 
 
 
lamodel is a network simulator that simulates memory engram formation 
in a population consisting of excitatory and inhibitory neurons with independent
dendritic subunits. 

This is the entry point for the network simulator. 
This file parses the command line parameters 

Basic command line options for the simulator:

Example: ./lamodel -P 2 -T 1400 [-G] [-L] [-s datadir-name] [-S random-seed]

-P <patterns>: number of memories  to enoode
-T <minutes>: Interval between memories (minutes)
-G: Global-only protein synthesis 
-L: Dendritic-only protein synthesis 
-s <dirname>: Name of the directory to store the data (inside the data/ folder)
-S <random-seed>: Intitialize the random generator

-o <param-name>=<param-value>: Set various simulation parameters:

-o nlTypePV=[0,1,2,3]   : Set basket cell dendrite nonlinearity type. 0=supra, 1=sub, 2=linear, 3=mixed supra/sub
-o nlTypeSOM=[0,1,2,3]  : Set SOM+ cell dendrite nonlinearity type. 0=supra, 1=sub, 2=linear, 3=mixed supra/sub
-o INClustered=[0,1]    : Set whether IN synapses  should target all dendrites randomly (dispersed) or only 33% of them (clustered)


*/


#include "constructs-nikos.h"
#include <iostream>
#include <cstring>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>
#include <sstream> 

using namespace std;


//*********************************MNIST LOADING FUNCTIONS*******************************************

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist(string filename, vector<vector<double> > &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        file.read((char*) &n_rows, sizeof(n_rows));
        n_rows = ReverseInt(n_rows);
        file.read((char*) &n_cols, sizeof(n_cols));
        n_cols = ReverseInt(n_cols);
        for(int i = 0; i < number_of_images; ++i)
        {
            vector<double> tp;
            for(int r = 0; r < n_rows; ++r)
            {
                for(int c = 0; c < n_cols; ++c)
                {
                    unsigned char temp = 0;
                    file.read((char*) &temp, sizeof(temp));
                    tp.push_back((double)temp);
                }
            }
            vec.push_back(tp);
        }
    }
}




void read_Mnist_Label(string filename, vector<double> &vec)
{
    ifstream file (filename, ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char*) &magic_number, sizeof(magic_number));
        magic_number = ReverseInt(magic_number);
        file.read((char*) &number_of_images,sizeof(number_of_images));
        number_of_images = ReverseInt(number_of_images);
        for(int i = 0; i < number_of_images; ++i)
        {
            unsigned char temp = 0;
            file.read((char*) &temp, sizeof(temp));
            vec[i]= (double)temp;
        }
    }
}



//*********************************MNIST LOADING FUNCTIONS^*******************************************


inline void program_input2(nrn_list lst, int tstart, int duration, float freq, float randomness, int limitActive = -1)
{
	int skip = 0;
	if (limitActive == -999)
		skip = 3;
	for (nrn_iter n = lst.begin(); n != lst.end(); ++n)
	{
		if (skip>0)
		{
			skip--;
			continue;
		}

		LAInput* in = (LAInput*)(*n);
		in->Program(tstart, duration, freq, randomness);
		//if (limitActive >0 && ++tot >= limitActive) return;
	}
}



void RunPattern(LANetwork& net,  vector<double> pat, double label, int active_nrn_spike_thresh, float max_inp_freq, float label_inp_freq, bool Train)
{

	int pad = 100;//time before and after pattern presentation, relaxing time for network
	int duration = 3800;//pattern presentation duration

	float frequency;//Instead of using 0s and 1s and standard frequency, I transform frequency based on pixel intensity.

	int i;



	for (i=0; i < 784; i++)//changed to MNIST vector size
	{
		LAInput* in = (LAInput*) net.pattern_inputs[i];
		frequency = max_inp_freq*pat[i]/255;//calc freq per pixel/input neuron
		in->Program(pad, duration, frequency, 0); // changed to frequency
	}
	if (Train)
	{
		LAInput* in = (LAInput*) net.label_inputs[label][0];
		in->Program(pad, duration, label_inp_freq, 0);
	}



	

	//program_input2(net.bg_list, pad, duration, .5, 0.5, -1);// encodes background noise neurons, let's not use it for starters. I could remodel this to be the specific subpopulation input

	//resets spike counters and calcium levels before StimDynamics
	net.ResetSpikeCounters();

	for (syn_iter si = net.synapses.begin(); si!= net.synapses.end(); si++)
		(*si)->calcium =0.;

	net.StimDynamics(duration+pad+pad);//some changes probably need to happen in this one to add second input population, JK ACTUALLY NOT!
	
	int tActive =0;//total active pyramidals
	int tSpikes =0;//total spikes
	int tActiveSub[10]={};// total spikes per subpopulation



	for (nrn_iter ni = net.pyr_list.begin(); ni != net.pyr_list.end(); ni++)
	{
		LANeuron* n = *ni;
		if (float(n->total_spikes) > active_nrn_spike_thresh )// if total spikes exceed active_nrn_spike_thresh the neuron is considered active
		{
			tActive++;
			//Draft
			tActiveSub[int(n->input_id)] += 1;
		}
		tSpikes += n->total_spikes;
	}


	for (i=0; i < 10; i++)
	{
		net.totPyrActiveSub[i] = tActiveSub[i];
	}
	net.totPyrActive = tActive;
	net.Prediction = distance(tActiveSub,max_element(tActiveSub,tActiveSub+10));



	cout<< "Subpopulation with most active neurons: " <<net.Prediction<<endl;
	cout<< "Max Active Neurons in subpopulation: " <<*max_element(tActiveSub,tActiveSub+10)<<endl;

	printf("Active pyrs= %d (%.2f%%), mean ff= %.2f\n", tActive, 100.0*float(tActive)/float(net.pyr_list.size()), tSpikes/(float(net.pyr_list.size())*2));
}



// Print out a vector of type T with spaces in between
template <typename T> static void PrintVector( vector<T>&  ar, ostream& outfile) 
{
	for (typename vector<T>::iterator it = ar.begin(); it != ar.end(); it++)
	{
		outfile << *it << ' ';
	}
	outfile << std::endl;
}



//Main function
//adding binary classification variables here too(also in CreateFearNet)
//turnover_iter ->Amount of training iterations between synaptic turnover events
//rewired_perc -> percentile of pruned synapses to be rewired during turnover
//turnover_thresh -> weight threshold for synaptic turnover
//adding code for loading csv centroid files
void RunSNN(LANetwork& net, int TOT_PAT_TRAIN, int TOT_PAT_TEST , string suffix, int EPOCHS, int turnover_iter, float rewired_perc, float turnover_thresh, int active_neuron_spike_thresh,
 float max_input_freq, float label_input_freq, float stop_thresh, float wmin, float wmax, int class_1, int class_2, int loadnetwork, int turnover_type)
{


//***********************TRAIN************************

    //read MNIST train images
    string train_img = "./MNIST/train-images-idx3-ubyte";
    int number_of_images_train = 60000;
    
    vector<vector<double> > train_imgvec;//contains the 60000 MNIST images for training
    read_Mnist(train_img, train_imgvec);


    //read MNIST train labels into double vector
    string train_labels = "./MNIST/train-labels-idx1-ubyte";

    vector<double> train_labelsvec(number_of_images_train);//contains the 60000 MNIST training image labels
    read_Mnist_Label(train_labels, train_labelsvec);


//******************CENTROID_TRAIN*******************

/*
vector<vector<double> > train_imgvec = read_csv("MNIST_centroid_images");

vector<vector<double>> templabs = read_csv("MNIST_centroid_labels");
vector<double> train_labelsvec = templabs[0];
*/

//***************************************************

//***********************TEST************************


    //read MNIST test images
    string test_img = "./MNIST/t10k-images-idx3-ubyte";
    int number_of_images_test = 10000;


    vector<vector<double> > test_imgvec;//contains the 10000 MNIST images for testing
    read_Mnist(test_img, test_imgvec);

    //read MNIST test labels into double vector
    string test_labels = "./MNIST/t10k-labels-idx1-ubyte";

    vector<double> test_labelsvec(number_of_images_test);//contains the 10000 MNIST testing image labels
    read_Mnist_Label(test_labels, test_labelsvec);

//***************************************************





//***********Index vector for suffling MNIST************

    vector<int> indexes;
	indexes.reserve(train_imgvec.size());
	for (unsigned long int i = 0; i < train_imgvec.size(); ++i)
    	indexes.push_back(i);
	random_shuffle(indexes.begin(), indexes.end());


//******************************************************


    int i,j,l; // loop variables
    int k=0;// training and testing loop counter
    int ii=0;// sample testing loop counter
    int sample_size=100;//sample testing batch size 
    int samp_tot=0;//counter used for sample accuracy
    int sample_interval=20;//Amount of training iterations between sample tests 
    int purged = 6;// variable used to identify how many synapses where purged during turnover
    int purged_1 = 6;// variable used to identify how many synapses where purged during turnover in subpop 1
    int purged_2 = 6;// variable used to identify how many synapses where purged during turnover in subpop 2   
    //int continualswitch = 100; //how many iterations until it switches to the other class
    int breakpoint= 350;// When to stop training
    float syn_counter; // counts synapses during stopping criterion loop
    float lr_counter; // counts synapses with lr < thresh during stopping criterion loop
    float lr_perc;	//percentile of synapses with lr < thresh during stopping criterion loop
    bool correct;
    //sequencially:
    int temp_class;//intermediate variable for switching between presented classes during sequencial image presentation
    int cls_1=class_1; //memory of original class 1
    int cls_2=class_2;//memory of original class 2
    int seq_len=1; //length of sequence of images in a class before switching, when presenting them sequencially and not randomly


	//ofstream traincnt(("./results/train_spikes_"+suffix+".txt").c_str());
	//ofstream pyrActiveCnt(("./results/pyr_active_train_"+suffix+".txt").c_str());

    //**********************COMMENT OUT FOR CLUSTER***********************
    
	ofstream pyrweightsperepochcnt(("./results/weights_per_epoch_"+suffix+".txt").c_str());
	ofstream pyrweightsperepochsub1cnt(("./results/weights_per_epoch_sub1"+suffix+".txt").c_str());
	ofstream pyrweightsperepochsub2cnt(("./results/weights_per_epoch_sub2"+suffix+".txt").c_str());
	
	//**********************COMMENT OUT FOR CLUSTER***********************
	
	ofstream accuracycnt(("./results/sample_accuracies_"+suffix+".txt").c_str());

	//ofstream presynnrn1cnt(("./results/presyn_to_nrn_sub1_"+suffix+".txt").c_str());
	//ofstream presynbr1cnt(("./results/presyn_to_br_sub1_"+suffix+".txt").c_str());
	ofstream presynw1cnt(("./results/presyn_w_sub1_"+suffix+".txt").c_str());
	//ofstream presynnrn2cnt(("./results/presyn_to_nrn_sub2_"+suffix+".txt").c_str());
	//ofstream presynbr2cnt(("./results/presyn_to_br_sub2_"+suffix+".txt").c_str());
	ofstream presynw2cnt(("./results/presyn_w_sub2_"+suffix+".txt").c_str());


	//ofstream inputcnt((("./results/input_data_"+suffix+".txt").c_str()));


	//synapse table checkup files 
	//ofstream syntabtraincnt(("./results/synaptic_table_train_"+suffix+".txt").c_str());
	


	//Syn Table Headers
	//syntabtraincnt<< "INPUT_IDX" << "	" << "INPUT_FR" << "	" << "HIDDEN_IDX" << "	" << "HIDDEN_FR" << "	"<< "HIDDEN_ACTIVE"
	//<< "	"<< "HIDDEN_SUBPOP" <<"	" << "WEIGHT" << "	" << "ITERATION" << "	" << "LABEL" << "	" << endl;

	if (loadnetwork ==1)
	{
		EPOCHS=0;
		cout<<"Network Loaded, testing now"<< endl;
	}


	net.enablePlasticity = true;
	for (j=0; j < EPOCHS; j++)
	{
		cout<<"EPOCH "<< j+1 << endl;
		for (i=0; i < TOT_PAT_TRAIN; i++)
		{

			//**************PRESENTING IMAGES RANDOMLY**************
			/*
			if (train_labelsvec[indexes[i]] != class_1 && train_labelsvec[indexes[i]] != class_2)
			{
				continue;
			}
			*/
			//*****************************************************
			
			//**********************************PRESENTING IMAGES SEQUENCIALLY**********************************
			
			if (train_labelsvec[indexes[i]] != class_1)
			{
				continue;
			}
			
			//**************************************************************************************************

			k+=1;

			//weight saving loop
			

			//**********************COMMENT OUT FOR CLUSTER***********************
			

			for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
			{
				LANeuron* nrn = *ni;
				for (syn_iter si = nrn->outgoing.begin(); si!=nrn->outgoing.end(); si++)
				{	
					pyrweightsperepochcnt<<(*si)->weight<<"	";
				}

			}
			pyrweightsperepochcnt << endl;
			


//*******************************PLAYING HERE WITH WEIGHT PER SUBPOPULATION**********************************************************************************************
			
			for (nrn_iter ni = net.pyr_lists[class_1].begin(); ni != net.pyr_lists[class_1].end(); ni++)
			{
				LANeuron* nrn = *ni;
				for (syn_iter si = nrn->incoming.begin(); si!=nrn->incoming.end(); si++)
				{	
					if (find(net.pattern_inputs.begin(),net.pattern_inputs.end(),(*si)->source_nrn) != net.pattern_inputs.end())
					{
						pyrweightsperepochsub1cnt<<(*si)->weight<<"	";
					}
					
				}

			}
			pyrweightsperepochsub1cnt << endl;

			
			for (nrn_iter ni = net.pyr_lists[class_2].begin(); ni != net.pyr_lists[class_2].end(); ni++)
			{
				LANeuron* nrn = *ni;
				for (syn_iter si = nrn->incoming.begin(); si!=nrn->incoming.end(); si++)
				{	
					if (find(net.pattern_inputs.begin(),net.pattern_inputs.end(),(*si)->source_nrn) != net.pattern_inputs.end())
					{
						pyrweightsperepochsub2cnt<<(*si)->weight<<"	";
					}
					
				}

			}
			pyrweightsperepochsub2cnt << endl;


			
			//**********************COMMENT OUT FOR CLUSTER***********************

			
//********************************************************************************************************************************************************************


			//**********************RESETS VECTORS*****************************
			for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
			{
				(*i)->spikeTimings.clear();
				(*i)->prpTransients.clear();
			}
			//*****************************************************************

			net.ResetSpikeCounters();
			cout << "Running " << k << " training pattern with label "<< train_labelsvec[indexes[i]]<< endl;

			RunPattern(net, train_imgvec[indexes[i]],train_labelsvec[indexes[i]],active_neuron_spike_thresh, max_input_freq, label_input_freq, true);

			
			//*****************INPUT SPIKE TIMES COLLECTING*********************
			
			
			//PrintVector<double>( train_imgvec[i] , inputcnt);
			/*
			for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
			{
				LANeuron* nrn = *ni;

				//Getting input spike times!
				PrintVector<int>( nrn->spikeTimings, inputcnt);
			}
			//inputcnt<<"end"<<endl;
			*/
			
			//******************************************************************
			


			/*
			for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
			{
				traincnt<< (*i)->total_spikes << " ";
			}
			traincnt <<endl;
			*/

			//pyrActiveCnt << net.totPyrActive << endl;

			//*************PRESENTING IMAGES RANDOMLY*************
			/*
			cout << "Interstim "  << endl ;
			net.Interstim(8280);
			*/
			//****************************************************

			//**********************************PRESENTING IMAGES SEQUENCIALLY**********************************
			
			
			if (k % seq_len == 0)
			{

				cout << "Interstim "  << endl ;
				net.Interstim(8280);
				//net.Interstim(7200);
				temp_class=class_1;
				class_1=class_2;
				class_2=temp_class;
				cout<<endl;
				
			}	

			else
			{
				cout << "Tagging..."  << endl ;
				net.Interstim(0);
			}
			
			
			//**************************************************************************************************

			//*********TEST ACCURACY IN SMALL SAMPLE************
			if ((k % sample_interval)==0)
			{
				//sample_interval = 50;
				cout<<"Running sample testing..."<< endl;
				cout<<endl;
				net.enablePlasticity = false;
				samp_tot=0;

				//******************************PRESYNAPTIC ANALYSIS FILES**************************

				for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
				{

					LANeuron* nrn = *ni;

					//presynnrn1cnt << (nrn->nid - 120);
					//presynbr1cnt << (nrn->nid - 120);
					presynw1cnt << (nrn->nid - 120);

					for (syn_iter si = nrn->outgoing.begin(); si != nrn->outgoing.end(); si++)		
					{

						LASynapse* s = *si;
						if (find(net.pyr_lists[class_1].begin(),net.pyr_lists[class_1].end(),s->target_nrn) != net.pyr_lists[class_1].end())
						{
							//presynnrn1cnt <<"	" << s->target_nrn->nid;
							//presynbr1cnt <<"	" << s->target_branch->bid;
							presynw1cnt <<"	" << s->weight;
						}

					}
					//presynnrn1cnt<<endl;
					//presynbr1cnt<<endl;
					presynw1cnt<<endl;

				}


				for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
				{

					LANeuron* nrn = *ni;

					//presynnrn2cnt << (nrn->nid - 120);
					//presynbr2cnt << (nrn->nid - 120);
					presynw2cnt << (nrn->nid - 120);

					for (syn_iter si = nrn->outgoing.begin(); si != nrn->outgoing.end(); si++)		
					{

						LASynapse* s = *si;
						if (find(net.pyr_lists[class_2].begin(),net.pyr_lists[class_2].end(),s->target_nrn) != net.pyr_lists[class_2].end())
						{
							//presynnrn2cnt <<"	" << s->target_nrn->nid;
							//presynbr2cnt <<"	" << s->target_branch->bid;
							presynw2cnt <<"	" << s->weight;
						}

					}
					//presynnrn2cnt<<endl;
					//presynbr2cnt<<endl;
					presynw2cnt<<endl;

				}


				//****************************************************************************




				for (l=0; l < TOT_PAT_TEST; l++)
				{
					
					//********CONDITION FOR BINARY CLASSIFICATION**********
					if (test_labelsvec[l] != class_1 && test_labelsvec[l] != class_2)
					{
						continue;
					}
					//*****************************************************

					ii++;

					//**********************RESETS VECTORS*****************************
					for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
					{
						(*i)->spikeTimings.clear();
						//(*i)->prpTransients.clear();
					}
					//*****************************************************************
					
					net.ResetSpikeCounters();
					

					cout << "Running " << ii << " sample testing pattern with label "<< test_labelsvec[l]<< endl;


					RunPattern(net,test_imgvec[l], 0, active_neuron_spike_thresh, max_input_freq, label_input_freq, false); //label set to 0 because it doesn't matter
					
					correct=(test_labelsvec[l]==net.Prediction);

					samp_tot+= int(correct);
					
					cout << endl ;

					if (ii == sample_size)
					{
						break;
					}

				}

				ii=0;
				net.enablePlasticity = true;
				accuracycnt << 100*samp_tot/sample_size << endl;

				cout<<"Sample Accuracy ="<< 100*samp_tot/sample_size << "%" << endl;
				cout<<endl;
			}
			//*************************************************

			//****************RANDOM SYNAPSE TURNOVER*****************
			if (turnover_type==0)
			{
				if ((k % turnover_iter)==0 && purged > 5)//Synaptic Turnover
				{
					purged = net.PurgeInputSynapsesNikos(wmax + turnover_thresh);
					int syns_kept = round(rewired_perc * purged);
					net.ConnectNeurons(net.pattern_inputs, net.pyr_list, 0, syns_kept, 1, wmin, wmax, false, true);
				}
			}
			
			//********************************************************




			//****************CONSTRAINED SYNAPSE TURNOVER*****************
			if (turnover_type==1)
			{
				
				if ((k % turnover_iter)==0 && purged_1 > 5 && purged_2 > 5)//Synaptic Turnover
				{
					cout<<"Turnover for 1st subpopulation"<<endl;
					purged_1 = net.PurgeInputSynapsesInSubpopNikos(wmax + turnover_thresh, net.pyr_lists[class_1]);
					int syns_kept_1 = round(rewired_perc * purged_1);
					net.ConnectNeurons(net.pattern_inputs, net.pyr_lists[class_1], 0, syns_kept_1, 1, wmin, wmax, false, true);

					
					cout<<"Turnover for 2nd subpopulation"<<endl;
					purged_2 = net.PurgeInputSynapsesInSubpopNikos(wmax + turnover_thresh, net.pyr_lists[class_2]);
					int syns_kept_2 = round(rewired_perc * purged_2);
					net.ConnectNeurons(net.pattern_inputs, net.pyr_lists[class_2], 0, syns_kept_2, 1, wmin, wmax, false, true);
					
				}
				
			}

			//*************************************************************



			//****************************EXTRA THINGY FOR CONTINUAL LEARNING STUFF*****************************
			/*
			if ((k == continualswitch))
			{
				purged = net.PurgeInputSynapsesNikos(0.5);
				
				int syns_kept = purged;
				
				net.ConnectNeurons(net.pattern_inputs, net.pyr_list, 0, syns_kept, 1, wmin, wmax, false, true);
			}
			*/
			//**************************************************************************************************


			//Syn Table Saving Loop
			/*
			cout<< "Saving Training Synaptic Table..." << endl;
			for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
			{
				LANeuron* nrn = *ni;
				for (syn_iter si = nrn->outgoing.begin(); si!=nrn->outgoing.end(); si++)
				{	
					syntabtraincnt<< (nrn->nid) - 250 
					 << "	" 
					 << round(nrn->total_spikes * 1000 / 3800 ) 
					 << "	"
					<< (*si)->target_nrn->nid 
					<< "	"  
					<< round((*si)->target_nrn->total_spikes * 1000 / 3800 ) 
					<< "	" 
					<< (float((*si)->target_nrn->total_spikes) /4. > 5.)
					<< "	" 
					<< (*si)->target_nrn-> input_id
					<< "	"
					<< (*si)->weight 
					<< "	" 
					<< k
					<< "	" 
					<< train_labelsvec[indexes[i]] 
					<< "	" 
					<< endl;
				}
			}			
			cout<< "Synaptic Training Table Saved!" << endl;
			cout<<endl;
			*/

			//Ending the loop after breakpoint successful iterations
			
			
			if (k==breakpoint)
			{
				cout<<"Hit the breakpoint,training finished"<<endl;
				break;
			}
			
			

			//****************************Stopping Criterion********************************************


			syn_counter = 0;
			lr_counter = 0;
			lr_perc = 0;

			for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
			{
				LANeuron* nrn = *ni;
				for (syn_iter si = nrn->outgoing.begin(); si!=nrn->outgoing.end(); si++)
				{	
					syn_counter+=1;
					if ((*si)->syn_learning_rate < ( (net.maxlr + net.minlr)/2 ))
					{
						lr_counter+=1;
					}	
				}
			}
			lr_perc = lr_counter/syn_counter;

			if (lr_perc >= stop_thresh)
			{	
				cout<< "***TRAINING FINISHED. TESTING NOW.***"<<endl;
				//purged = net.PurgeInputSynapsesNikos(wmax + turnover_thresh);
				break;
			}
		}
		if (lr_perc >= stop_thresh)
			{
				break;
			}
			//******************************************************************************************		
	}


//**********************COMMENT OUT FOR CLUSTER***********************

	for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
	{
		LANeuron* nrn = *ni;
		for (syn_iter si = nrn->outgoing.begin(); si!=nrn->outgoing.end(); si++)
		{	
			pyrweightsperepochcnt<<(*si)->weight<<"	";
		}
	}
	pyrweightsperepochcnt << endl;

//**********************COMMENT OUT FOR CLUSTER***********************	
	//************************SAVING NETWORK********************************************************
	if (loadnetwork == 0)
	{
		cout<<"Saving Network..."<<endl;
		ofstream netcnt(("./results/network_state_"+suffix+".txt").c_str());// 
		for (syn_iter si =net.synapses.begin(); si != net.synapses.end(); ++si)
		{
			LASynapse* s = *si;
			netcnt << s->source_nrn->nid << "," << s->target_branch->bid << "," << s->weight << endl;
		}
	}
	//***********************************************************************************************



//WEIGHT SAVING FILE(ALL WEIGHTS)
	//ofstream weightcnt(("train_weights_"+suffix+".txt").c_str());

	//for (syn_iter si = net.synapses.begin(); si!= net.synapses.end(); si++)
	//	weightcnt << (*si)->weight << " ";
	//weightcnt<<endl;


	//******************************SYNAPSE LOCATION ANALYSIS FILE**************************
	ofstream synclustcnt(("./results/synapse_analysis_"+suffix+".txt").c_str());
	for (nrn_iter ni = net.pyr_list.begin(); ni != net.pyr_list.end(); ni++)
	{
		LANeuron* nrn = *ni;
		for (branch_iter bi = nrn->branches.begin(); bi != nrn->branches.end(); ++bi)
		{
			LABranch* b = *bi;

			synclustcnt << nrn->nid << "	" << b->bid << "	"<< nrn->input_id << "	"<< b->synapses.size();
			for (syn_iter si = b->synapses.begin(); si != b->synapses.end(); si++)		
			{
				LASynapse* s = *si; 
				if (find(net.pattern_inputs.begin(),net.pattern_inputs.end(),s->source_nrn) != net.pattern_inputs.end())
				{
					synclustcnt<< "	" << s->weight;
				
				}
			
			}
			synclustcnt << endl;
		}
	}

	//****************************************************************************

	//******************************PRESYNAPTIC ANALYSIS FILES**************************
	class_1=cls_1;
	class_2=cls_2;

	for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
	{

		LANeuron* nrn = *ni;

		//presynnrn1cnt << (nrn->nid - 120);
		//presynbr1cnt << (nrn->nid - 120);
		presynw1cnt << (nrn->nid - 120);

		for (syn_iter si = nrn->outgoing.begin(); si != nrn->outgoing.end(); si++)		
		{

			LASynapse* s = *si;
			if (find(net.pyr_lists[class_1].begin(),net.pyr_lists[class_1].end(),s->target_nrn) != net.pyr_lists[class_1].end())
			{
				//presynnrn1cnt <<"	" << s->target_nrn->nid;
				//presynbr1cnt <<"	" << s->target_branch->bid;
				presynw1cnt <<"	" << s->weight;
			}

		}
		//presynnrn1cnt<<endl;
		//presynbr1cnt<<endl;
		presynw1cnt<<endl;

	}


	for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
	{

		LANeuron* nrn = *ni;

		//presynnrn2cnt << (nrn->nid - 120);
		//presynbr2cnt << (nrn->nid - 120);
		presynw2cnt << (nrn->nid - 120);

		for (syn_iter si = nrn->outgoing.begin(); si != nrn->outgoing.end(); si++)		
		{

			LASynapse* s = *si;
			if (find(net.pyr_lists[class_2].begin(),net.pyr_lists[class_2].end(),s->target_nrn) != net.pyr_lists[class_2].end())
			{
				//presynnrn2cnt <<"	" << s->target_nrn->nid;
				//presynbr2cnt <<"	" << s->target_branch->bid;
				presynw2cnt <<"	" << s->weight;
			}

		}
		//presynnrn2cnt<<endl;
		//presynbr2cnt<<endl;
		presynw2cnt<<endl;

	}


	//****************************************************************************


	
	//*************Reset CREB levels and close palsticity*****************
	net.enablePlasticity = false;

	for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
		{
			(*i)->crebLevel=0.0;
		}
	//********************************************************************
	//ofstream testcnt(("./results/test_spikes_"+suffix+".txt").c_str());
	//ofstream pyrActiveTestCnt(("./results/pyr_active_test_"+suffix+".txt").c_str());
	ofstream predictions(("./results/predictons_"+suffix+".txt").c_str());
	//ofstream testactivitycnt(("./results/test_activity_"+suffix+".txt").c_str());
	//ofstream ActivePerSub(("./results/pyr_active_per_sub_test_"+suffix+".txt").c_str());
	//ofstream syntabtestcnt(("./results/synaptic_table_test_"+suffix+".txt").c_str());
	//ofstream testspikescnt(("./results/test_spike_table_"+suffix+".txt").c_str());
	//Syn Table Headers
	//syntabtestcnt<< "INPUT_IDX" << "	" << "INPUT_FR" << "	" << "HIDDEN_IDX" << "	" << "HIDDEN_FR" << "	"<< "HIDDEN_ACTIVE"
	//<< "	"<< "HIDDEN_SUBPOP" <<"	" << "WEIGHT" << "	" << "ITERATION" << "	" << "LABEL" << "	" << endl;

	k=0;

	for (i=0; i < TOT_PAT_TEST; i++)
	{
		
		//********CONDITION FOR BINARY CLASSIFICATION**********
		if (test_labelsvec[i] != class_1 && test_labelsvec[i] != class_2)
		{
			continue;
		}
		//*****************************************************

		k++;


		//**********************RESETS VECTORS*****************************
		for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
		{
			(*i)->spikeTimings.clear();
			//(*i)->prpTransients.clear();
		}
		//*****************************************************************

		net.ResetSpikeCounters();

		cout << "Running " << k << " testing pattern with label "<< test_labelsvec[i]<< endl;
		RunPattern(net,test_imgvec[i], 0, active_neuron_spike_thresh, max_input_freq, label_input_freq, false); // label set to 0 because it doesn't matter
		
		correct=(test_labelsvec[i]==net.Prediction);


		

		/*
		for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
		{
			LANeuron* nrn = *ni;

			//Getting input spike times!
			PrintVector<int>( nrn->spikeTimings, inputcnt);
		}
		*/

	


		/*
		for (nrn_iter i = net.neurons.begin(); i != net.neurons.end(); i++)
		{
			testcnt<< (*i)->total_spikes << " " ;
			//cout << (*i)->total_spikes << " " ;
		}
		testcnt <<endl;
		cout << endl ;
		*/

		//pyrActiveTestCnt << net.totPyrActive << endl;


		//************ACTIVE PER SUB****************

		/*
		for (j=0; j < 10; j++)
		{
			ActivePerSub << net.totPyrActiveSub[j] << " ";
		} 
		ActivePerSub << endl;

		*/

		//*******************************************


		predictions<< test_labelsvec[i]<<" "<< net.Prediction << " " << correct << endl;
		

		//*******************SPIKE TABLE FOR POST HOC ANALYSIS***********************

		/*
		testspikescnt << "SUBPOP" << "	" << "SPIKES" << endl;
		for (nrn_iter ni = net.pyr_list.begin(); ni != net.pyr_list.end(); ni++)
		{
			LANeuron* nrn = *ni;
			testspikescnt << nrn->input_id << "	" << nrn -> total_spikes << endl;
		}
		testspikescnt << endl;
		*/
		
		//***************************************************************************


		//************PULLING TABLE FOR MEAN PER CLASS ACTIVITY PER NODE(NEURONS AND DENDRITES) IN TEST SET**************
		/*
		testactivitycnt<< "NEURONID" << "	" << "NEURON_SPIKES" << "	" << "BRANCHID" << "	" << "BRANCH_SPIKES" << "	" << "SUBPOP" << "	" << "LABEL" << "	" << "PREDICTION" << endl;
		for (nrn_iter ni = net.pyr_list.begin(); ni != net.pyr_list.end(); ni++)
		{
			LANeuron* nrn = *ni;
			for (branch_iter bi = nrn->branches.begin(); bi != nrn->branches.end(); ++bi)
			{
				LABranch* b = *bi;

				testactivitycnt << nrn->nid << "	" << nrn ->total_spikes << "	" << b->bid << "	" << b->branch_spikes << "	" << nrn->input_id << "	" << test_labelsvec[i] << "	" << net.Prediction << endl;
			}
		}
		testactivitycnt<<endl;
		*/

		//***************************************************************************************************************

		/*
		cout<< "Saving Testing Synaptic Table..." << endl;
		for (nrn_iter ni = net.pattern_inputs.begin(); ni != net.pattern_inputs.end(); ni++)
		{
			LANeuron* nrn = *ni;
			for (syn_iter si = nrn->outgoing.begin(); si!=nrn->outgoing.end(); si++)
			{	
				syntabtestcnt<< (nrn->nid) - 250 
				 << "	" 
				 << round(nrn->total_spikes * 1000 / 3800 ) 
				 << "	"
				<< (*si)->target_nrn->nid 
				<< "	"  
				<< round((*si)->target_nrn->total_spikes * 1000 / 3800 ) 
				<< "	" 
				<< (float((*si)->target_nrn->total_spikes) /4. > 5.)
				<< "	" 
				<< (*si)->target_nrn-> input_id
				<< "	"
				<< (*si)->weight 
				<< "	" 
				<< k
				<< "	" 
				<< test_labelsvec[i] 
				<< "	" 
				<< endl;
			}
		}			
		cout<< "Synaptic Testing Table Saved!" << endl;
		*/
		cout<<endl;
	}
}




// Parse command line  arguments
int main( int argc, char* argv[])
{
	
	int c;

	// Set defaults
	int nneurons = 100;
	int nbranches = 10;
	//int ninputs = 800;
	//int nperinput = 1;
	//int npatterns = ninputs;
	//int nonesperpattern = 1;
	//int interstim = 60;
	int rseed = 1980;
	char* suffix = NULL;
	bool disableCreb = false;

	//Nikos Stuff
	int numPatternsTrain = 60000;//60000 for full mnist, 1006 for reduced centroid mnist
	int numPatternsTest = 10000; 
	int epochs=1;
	int digit1 = 0;
	int digit2 = 1;
	float inh_control_synapses = 500;
	float pattern_to_hidden_synapses= 1750;
	float label_to_hidden_synapses = 2;
	float lateral_inh_synapses = 4;
	float Wmin = 0.1;
	float Wmax = 0.2;
	int iterations_to_turnover = 20;
	float rewired_percentile = 1;
	float turnover_threshold = 0;
	int active_neuron_spike_threshold = 20;
	float max_input_frequency = 25;
	float label_input_frequency= 40;
	float stop_threshold = 0.5;
	int load = 0;
	int turn_type = 1;//0 = Random , 1 = Constrained, 2= No Turnover



	// The network object
	LANetwork net; 


	net.nlTypePYR=0; //supra
	net.nlTypeSOM = 1;//sub
	net.nlTypePV = 0;//supra
	
	//net.enablePruning = true; 

	while ((c = getopt(argc, argv, "M:N:H:B:I:i:P:p:T:S:s:d:w:O:g:l:b:c:o:t:xnLDRJCGhU"))!= -1)
	{
		switch (c)
		{
			case '?':
			case 'h':
			cout <<
			"Basic command line options for the simulator:" << endl << 
			"Example: ./lamodel -P 2 -T 1400 [-G] [-L] [-s datadir-name] [-S random-seed]" << endl << 
			"-P <patterns>: number of memories  to enoode"<< endl << 
			"-T <minutes>: Interval between memories (minutes)"<< endl << 
			"-G: Global-only protein synthesis "<< endl << 
			"-L: Dendritic-only protein synthesis "<< endl << 
			"-s <dirname>: Name of the directory to store the data (inside the ./data/ folder)"<< endl << 
			"-S <random-seed>: Initialize the random generator"<< endl << 
			"-o <param-name>=<param-value>: Set various simulation parameters, for example:"<< endl << 
			"-o nlTypePV=[0,1,2,3]   : Set basket cell dendrite nonlinearity type. 0=supra, 1=sub, 2=linear, 3=mixed supra/sub"<< endl << 
			"-o nlTypeSOM=[0,1,2,3]  : Set SOM+ cell dendrite nonlinearity type. 0=supra, 1=sub, 2=linear, 3=mixed supra/sub" << endl << 
			"-o nlTypePYR=[0,1,2,3]  : Set pyramidal cell dendrite nonlinearity type. 0=supra, 1=sub, 2=linear, 3=mixed supra/sub" << endl <<
			"-o INClustered=[0,1]    : Set whether IN synapses  should target all dendrites randomly (dispersed) or only 33% of them (clustered)" << endl;
			return 0;
			break;

			case 'B': nbranches = atoi(optarg); break;
			//case 'I': ninputs = atoi(optarg); break;
			//case 'i': nperinput = atoi(optarg); break;
			case 'N': nneurons = atoi(optarg); break;
			//case 'P': npatterns = atoi(optarg); break;
			//case 'p': nonesperpattern = atoi(optarg); break;
			//case 'T': interstim = atoi(optarg); break;
			case 'S': rseed = ( atoi(optarg)); break;
			case 's': suffix = strdup(optarg); break;

			case 'n': disableCreb = true; break;
			case 'w': net.isWeakMem.push_back(atoi(optarg)-1); break;

			case 'L': net.localProteins = true; break;
			case 'G': net.globalProteins = true; break;
			case 'D': net.debugMode = true; break;
			case 'R': net.repeatedLearning = true; break;
			case 'J': net.pretraining = true; break;
			case 'C': net.altConnectivity = true; break;
			case 'O': net.branchOverlap = atof(optarg); break;
			case 'H': net.homeostasisTime = atof(optarg); break;


			case 'o': 
				char* o = strstr(optarg, "=");
				if (o)
				{
					*o = '\0';
					char* val = o+1;

					if (!strcmp(optarg, "connectivityParam")) net.connectivityParam = atof(val); 
					else if (!strcmp(optarg,  "BSPTimeParam")) net.BSPTimeParam = atof(val); 
					else if (!strcmp(optarg,  "homeostasisTimeParam")) net.homeostasisTimeParam = atof(val); 
					else if (!strcmp(optarg,  "CREBTimeParam")) net.CREBTimeParam = atof(val); 
					else if (!strcmp(optarg,  "inhibitionParam")) net.inhibitionParam = atof(val); 
					else if (!strcmp(optarg,  "globalPRPThresh")) net.globalPRPThresh = atof(val); 
					else if (!strcmp(optarg,  "localPRPThresh")) net.localPRPThresh = atof(val); 
					else if (!strcmp(optarg,  "dendSpikeThresh")) net.dendSpikeThresh = atof(val); 
					else if (!strcmp(optarg,  "initWeight")) net.initWeight*= atof(val); 
					else if (!strcmp(optarg,  "maxWeight")) net.maxWeight*= atof(val); 
					else if (!strcmp(optarg,  "stimDurationParam")) net.stimDurationParam = atof(val); 
					else if (!strcmp(optarg,  "nNeuronsParam")) nneurons *= atof(val); 
					else if (!strcmp(optarg,  "nBranchesParam")) nbranches *= atof(val); 
					else if (!strcmp(optarg,  "nBranchesTurnover")) net.nBranchesTurnover = atoi(val); 
					else if (!strcmp(optarg,  "INClustered")) net.INClustered = atoi(val); 
					else if (!strcmp(optarg,  "nlTypePV")) net.nlTypePV = atoi(val); 
					else if (!strcmp(optarg,  "nlTypeSOM")) net.nlTypeSOM = atoi(val); 
					else if (!strcmp(optarg,  "nlTypePYR")) net.nlTypePYR = atoi(val); 
					else if (!strcmp(optarg,  "hasCaap")) net.hasCaap = atoi(val);
					//NIKOS HYPERPARAMETER ADDITION 
					else if (!strcmp(optarg,  "numPatternsTrain")) numPatternsTrain = atoi(val); 
					else if (!strcmp(optarg,  "numPatternsTest")) numPatternsTest = atoi(val);
					else if (!strcmp(optarg,  "digit1")) digit1 = atoi(val);
					else if (!strcmp(optarg,  "digit2")) digit2 = atoi(val);
					else if (!strcmp(optarg,  "inh_control_synapses")) inh_control_synapses = atof(val);
					else if (!strcmp(optarg,  "pattern_to_hidden_synapses")) pattern_to_hidden_synapses = atof(val);
					else if (!strcmp(optarg,  "label_to_hidden_synapses")) label_to_hidden_synapses = atof(val);
					else if (!strcmp(optarg,  "lateral_inh_synapses")) lateral_inh_synapses = atof(val);
					else if (!strcmp(optarg,  "Wmin")) Wmin = atof(val);
					else if (!strcmp(optarg,  "Wmax")) Wmax = atof(val);
					else if (!strcmp(optarg,  "slope")) net.lr_slope = atof(val);
					else if (!strcmp(optarg,  "midpos")) net.lr_midpos = atof(val);
					else if (!strcmp(optarg,  "maxlr")) net.maxlr = atof(val);
					else if (!strcmp(optarg,  "minlr")) net.minlr = atof(val);
					else if (!strcmp(optarg,  "iterations_to_turnover")) iterations_to_turnover = atoi(val);
					else if (!strcmp(optarg,  "rewired_percentile")) rewired_percentile = atof(val);
					else if (!strcmp(optarg,  "turnover_threshold")) turnover_threshold = atof(val);
					else if (!strcmp(optarg,  "active_neuron_spike_threshold")) active_neuron_spike_threshold = atoi(val);
					else if (!strcmp(optarg,  "max_input_frequency")) max_input_frequency = atof(val);
					else if (!strcmp(optarg,  "label_input_frequency")) label_input_frequency = atof(val);
					else if (!strcmp(optarg,  "stop_threshold")) stop_threshold = atof(val);
					else if (!strcmp(optarg,  "LoadNet")) load = atoi(val);
					else if (!strcmp(optarg,  "Turnover_Type")) turn_type = atoi(val);

					printf("Parameter name='%s' value='%f'\n", optarg, atof(val));
				}
			break;
		}
	}


	LANetwork::SetRandomSeed(rseed);
	net.disableCreb = disableCreb;

	// Create the network connectivity
	if (load==0)
	{
		cout<<"Creating Network..."<<endl;
		net.CreateFearNet(nneurons, nbranches, inh_control_synapses, pattern_to_hidden_synapses, label_to_hidden_synapses, lateral_inh_synapses, Wmin, Wmax, digit1, digit2);
	}

	else if (load==1)
	{
		cout<<"Loading Network..."<<endl;
		net.LoadNet(nneurons, nbranches, digit1, digit2, "./results/network_state_asdf.txt");
	}

	// Set data directory from command line
	//char buf[512];
	//if (suffix)
	//	sprintf(buf, "./data/%s", suffix );
	//else // Default  data directory name
	//	sprintf(buf, "./data/N%d.B%d.I%d.i%d.P%d.p%d.T%d.S%d.w%d_%s", nneurons, nbranches, ninputs, nperinput, npatterns, nonesperpattern, interstim, rseed, (int)net.isWeakMem.size(),  suffix ? suffix : "");

	//cout << "Data output directory is "<< buf <<  endl;
	//net.SetDataDir( buf );

	LANetwork::SetRandomSeed(rseed);

	//srand(rseed+80);

	cout << "Running main simulation..."<< endl;
	cout << endl;
	RunSNN(net, numPatternsTrain, numPatternsTest, suffix, epochs, iterations_to_turnover, rewired_percentile, turnover_threshold,
	 active_neuron_spike_threshold, max_input_frequency, label_input_frequency, stop_threshold, Wmin, Wmax, digit1, digit2, load, turn_type); 
	return 0;

}

