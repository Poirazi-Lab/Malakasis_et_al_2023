/*

lamodel main implementation file. 
See constructs.h for class definitions

*/



#include "constructs-nikos-continual.h"
#include <iterator>
#include <assert.h>
#include <random>
#include <chrono>
#include <algorithm>

const int DEBUG_SID = 3024;
const int CUTOFF = 10.0;


#define VEC_REMOVE(vec, t) (vec).erase(std::remove((vec).begin(), (vec).end(), t), (vec).end())


// PROTEIN Synthesis thresholds. When calcium goes above this value, proteins will be synthesized
const float GPROD_CUTOFF = 18.0; //  For the case of Global protein synthesis
const float BPROD_CUTOFF = 1.8;  //  For the case of local (branch-specific) protein synthesis


int LANetwork::RSEED = 1980; // initial Random seed, to override with -S command line option 



// The alpha function that models protein sythesis over time (x is time in secs)
inline double nuclearproteinalpha(float x)
{
	return (x>1200)*((x-20.*60.)/(30.*60)) * exp(1. - (x-20.*60. )/(30.*60.));
}




// The alpha function for protein sythesis in dend branches over time (x is time in secs)
inline double branchproteinalpha(float x)
{
	return ((x)/(15.*60)) * exp(1. - (x )/(15.*60.));
}



// Returns the (positive/LTP or negative/LTD) magnitude of the synaptic tags depending on the  Calcium level ca
inline float caDP(float ca)
{


	float f = (1.3/(1.+exp(-(ca*10.-3.5)*10.))) - (0.3/(1.+exp(-(ca*10.0-2.0)*19.)));
	return f;
}



//  Create a random  spike train in a list of input neurons, starting at t=tstart after the start of StimDynamics (msec), duration is the length in msec , freq is the average spike frequency , randomness is uniform noise 
inline void program_input(nrn_list lst, int tstart, int duration, float freq, float randomness)
{
	for (nrn_iter n = lst.begin(); n != lst.end(); ++n)
	{
		LAInput* in = (LAInput*)(*n);
		in->Program(tstart, duration, freq, randomness);
	}
}




// Create a list of 'number' neurons, with 'n_branches_per_neuron' branches each, set their type to type 'type', and  them to the  vector 'appendTo'
// Set their input_id to 'inputId'
void LANetwork::CreateNeurons(int number, int n_branches_per_neuron, char type, vector<LANeuron*>* appendTo = 0, vector<LANeuron*>* appendTo2 = 0,  int inputId =-1)
{
	for (int i =0 ;i < number; i++)
	{
		LANeuron* n ;
		if (type == 'S') // Type = source neuron that just emits spikes (memory stimulation)
		{
			n = new LAInput();
			n->network = this;

			n->input_id = inputId;
			((LAInput*)n)->groupIdx = i;
		}
		else // Type = normal neuron that has branches and all
		{
			n = new LANeuron();
			n->network = this;
			n->input_id = inputId;
			for (int tt =0; tt < n_branches_per_neuron; tt++)
			{
				LABranch* bb  = new LABranch;
				bb->bid = this->branches.size();
				bb->neuron = n;


				// Set type of nonlinearity according to dendrite type and command line options
				if (type == 'P')  // Pyramidals
				{
					bb->nlType = DEND_SUPRA;
				}
				else if ( type == 'M') // SOM interneurons
				{
					if (nlTypeSOM == DEND_MIXED)
					{
						if (tt < 0.5*n_branches_per_neuron) bb->nlType = DEND_SUB;
						else bb->nlType = DEND_SUPRA;
					}
					else if (nlTypeSOM < DEND_MIXED && nlTypePV >= 0)
						bb->nlType = nlTypeSOM;
					else
					{
						printf("bad type %d", nlTypeSOM);
						abort();
					}
				}
				else if (type == 'V') // basket interneurons
				{
					if (nlTypePV == DEND_MIXED)
					{
						if (tt < 0.5*n_branches_per_neuron) bb->nlType = DEND_SUB;
						else bb->nlType = DEND_SUPRA;
					}
					else if (nlTypePV < DEND_MIXED && nlTypePV >= 0)
						bb->nlType = nlTypePV;
					else
					{
						printf("bad type %d", nlTypePV);
						abort();
					}

				}


				this->branches.push_back(bb);
				n->branches.push_back(bb);
			}
		}
		n->type = type;

		n->V = 1.0;
		n->nid = this->neurons.size();

		this->neurons.push_back(n);
		if (appendTo)
			appendTo->push_back(n);
		if (appendTo2)
			appendTo2->push_back(n);
	}
}




// Add a synaptic connection from Neuron 'a'  to target Neuron Branch 'br' with initial weight 'weight', and whether the synapse isPlastic or not
void LANetwork::AddSynapse(LANeuron* a, LABranch* br, float weight, bool isPlastic)
{
		LASynapse* syn = new LASynapse();
		syn->sid  = this->synapsesCounter++;
		syn->source_nrn = a;
		syn->target_nrn = br->neuron;
		syn->isPlastic = isPlastic;

		syn->weight  = weight;
		syn->target_branch = br; 
		if (isPlastic)
		{
			syn->syn_learning_rate = this->maxlr;
		}

		syn->source_nrn->outgoing.push_back(syn);
		syn->target_nrn->incoming.push_back(syn);
		syn->target_branch->synapses.push_back(syn);
		syn->pos = rgen();
		this->synapses.push_back(syn);
}



// Create a number of synaptic connections between two lists of neurons
// Connect random neurons from vector 'fromList' to random neurons in vector 'toList', 
// if isClustered , then the synapses will target only 1/3 of the target branches , simulating increased synapse density in fewer branches
// nNeuronPairs = how many synapses to add, nSynapsesPerNeuron = minimum number of synapses per pair

int LANetwork::ConnectNeurons(vector<LANeuron*> fromList, vector<LANeuron*> toList, bool isClustered,  int nNeuronPairs, int nSynapsesPerNeuron, float initialWeightMin = -1, float initialWeightMax = -1, bool fullCon=false, bool isPlastic= false)
{

	int tpairs =0;
	vector<LANeuron*> toList2;
	toList2.assign(toList.begin(),toList.end());
	

	unsigned seed = LANetwork::RSEED;
	std::uniform_real_distribution<float> distribution(initialWeightMin,initialWeightMax);
	std::mt19937_64 rng(seed);

	while(true)
	{
		if (initialWeightMin != -1)
		{
			float initialWeight = distribution(rng);
			initialWeightMax = initialWeight;
		}

		
		
		LANeuron* a = fromList.at(int(rgen()*(float)fromList.size()));
		int toIdx;
		LANeuron* b;

		if (fullCon)
		{
			toIdx = int(rgen()*(float)toList2.size());
			b = toList2.at(toIdx);
			toList2.erase(toList2.begin() + toIdx);

			if (toList2.size()==0)
			{
				toList2.assign(toList.begin(),toList.end());
			}
		}
		else
		{
			toIdx=int(rgen()*(float)toList.size());
			b = toList.at(toIdx);
		}


		for (int i =0; i < nSynapsesPerNeuron; i++)
		{

			float rval;
			if (isClustered)
				rval = rgen()*float(b->branches.size()/3);  // Clusterd
			else
				rval = rgen()*float(b->branches.size());

			LABranch* br =  b->branches[(int)rval];

			this->AddSynapse(a, br, initialWeightMax, isPlastic);	
		}

		if (++tpairs >= nNeuronPairs) break;
	}
	return tpairs;
}



// A random integer from 0 to max
inline int randomindex(int max)
{
	return (int)(rgen()*float(max));
}


// For Synaptic turnover simulation
// Remove synapses that are weaker than weightLimit

int LANetwork::PurgeInputSynapses(int totalToRemove,float weightLimit) // NIKOS: Changing inputs_cs with my pattern_inputs vector, since that one is not usable
{

	int totalFound =0;
	int totalTries = totalToRemove*3;
	while (totalFound < totalToRemove && totalTries-->0)
	{
		nrn_list lst = this->pattern_inputs;
		LANeuron* n = lst.at(randomindex(lst.size()));
		if (n->outgoing.size())
		{
			LASynapse* s = n->outgoing.at(randomindex(n->outgoing.size()));
			if (s->target_nrn->type == 'P'  && s->weight <= weightLimit)
			{
				//candidate for deletion

				//pthread_mutex_lock(&this->synapses_mutex);

				VEC_REMOVE(s->source_nrn->outgoing, s);
				VEC_REMOVE(s->target_nrn->incoming, s);
				VEC_REMOVE(s->target_branch->synapses, s);
				VEC_REMOVE(this->synapses, s);
				//cout << " Removing " << s->sid << endl;
				delete s; 

				//pthread_mutex_unlock(&this->synapses_mutex);

				totalFound++;
			}
		}
	}


	cout << "**********" << totalFound <<  " Synapses removed."  << "**********" << endl;  


	return totalFound;
}


//*************************************************************PLAYING AROUND HERE******************************************************

int LANetwork::PurgeInputSynapsesNikos(float weightLimit) // NIKOS: Changing inputs_cs with my pattern_inputs vector, since that one is not usable
{
	int totalFound =0;
	for (nrn_iter ni = this->pattern_inputs.begin(); ni != this->pattern_inputs.end(); ni++)
	{
		LANeuron* n = *ni;
		if (n->outgoing.size())
		{
			syn_iter si = n->outgoing.begin();

			while (si != n->outgoing.end())
			{
				LASynapse* s = *si;
				if (s->weight <= weightLimit)
				{
					VEC_REMOVE(s->source_nrn->outgoing, s);
					VEC_REMOVE(s->target_nrn->incoming, s);
					VEC_REMOVE(s->target_branch->synapses, s);
					VEC_REMOVE(this->synapses, s);
					delete s; 

					totalFound++;
				}
				else
				{
					si++;
				}
			}
		}
	}
//***********************************************************************************************************************************

	cout << "**********" << totalFound <<  " Synapses removed."  << "**********" << endl;  


	return totalFound;
}



// Create the network neurons and connectivity
//Adding two function options, in order to be able to run from terminal multiple binary classifications.
//pat_to_hid_syns -> from pattern input to hidden layer
//inh_ctrl_syns -> between inters and pyrams,with George multipliers
//lab_to_hid_syns -> from label input neuron to hidden neurons(multiplied by 0.5 for binary classification per subpop)
//lat_inh_syns -> from and to lateral inhibition interneurons

void LANetwork::CreateFearNet(int nneurons, int nbranches, float inh_ctrl_syns, float pat_to_hid_syns, float lab_to_hid_syns,

 float lat_inh_syns, float initWmin, float initWmax, int class_1, int class_2)
{
	this->n_neurons = nneurons;
	this->n_branches_per_neuron = nbranches;


	// Excitatory population (Pyr) (80% of total) (10% per digit class)
	this->CreateNeurons(this->n_neurons*0.8*0.5, this->n_branches_per_neuron, 'P', &this->pyr_lists[class_1], &this->pyr_list, class_1);
	this->CreateNeurons(this->n_neurons*0.8*0.5, this->n_branches_per_neuron, 'P', &this->pyr_lists[class_2], &this->pyr_list, class_2);
	
	//this->CreateNeurons(this->n_neurons*0.8*0.25, this->n_branches_per_neuron, 'P', &this->pyr_lists[class_3], &this->pyr_list, class_3);
	//this->CreateNeurons(this->n_neurons*0.8*0.25, this->n_branches_per_neuron, 'P', &this->pyr_lists[class_4], &this->pyr_list, class_4);


	// Inhibitory populations - PV+  (10%) 
	this->CreateNeurons(this->n_neurons*0.1, 10 , 'V', &this->in_pv); // SOM

	// Inhibitory populations - SOM+ (10%) 
	this->CreateNeurons(this->n_neurons*0.1, 10 , 'M', &this->in_som); // PV 

	/* 
	LATERAL INHIBITION
	*/

	this->CreateNeurons(this->n_neurons*0.1, 10 , 'V', &this->in_pv_0); 
	this->CreateNeurons(this->n_neurons*0.1, 10 , 'V', &this->in_pv_1); 
	//this->CreateNeurons(this->n_neurons*0.05, 10 , 'V', &this->in_pv_2); 
	//this->CreateNeurons(this->n_neurons*0.05, 10 , 'V', &this->in_pv_3); 

	// Create input stimulation neurons that bring in the patterns  CHANGING THAT TO 784 FOR MNIST
	CreateNeurons(784, 0, 'S', &this->pattern_inputs);


	// Create input stimulation neurons that fire according to the digit's label. 100 per digit class for starters
	CreateNeurons(1, 0, 'S', &this->label_inputs[class_1]);
	CreateNeurons(1, 0, 'S', &this->label_inputs[class_2]);
	//CreateNeurons(1, 0, 'S', &this->label_inputs[class_3]);
	//CreateNeurons(1, 0, 'S', &this->label_inputs[class_4]);
	


	// Create some Background noise inputs 
	//CreateNeurons(10, 0, 'S', &this->bg_list);


	// Add a bunch of synapses between the created population lists. 

	
	//float  baseSyns = 500 /*inh_ctrl_syns*/ ;

	// Pur <-> basket cells  connections
	ConnectNeurons(this->pyr_list, this->in_pv, 0,  1*inh_ctrl_syns, 1, -1, .3, false, false);//500


	ConnectNeurons(this->in_pv, this->pyr_list, 0, 1*10.*inh_ctrl_syns, 1, -1, .3, false, false);//5000



	// Pyr <-> SOM cells connections
	ConnectNeurons(this->pyr_list, this->in_som, 0,  0.1*2*inh_ctrl_syns, 1, -1, .3, false, false);//100


	ConnectNeurons(this->in_som, this->pyr_list, 0,  0.1*4*inh_ctrl_syns, 1, -1, .3, false, false);//400




	//baseSyns = 3500 /*pat_to_hid_syns*/ ;


	// Source neurons stimulation (memory patterns) -> Pyr  connections  (plastic)
	this->ConnectNeurons(this->pattern_inputs, this->pyr_list, 0, pat_to_hid_syns, 1, initWmin, initWmax , true, true);



	float LabelWeight = 1;
	//float labelSyns=n_neurons*0.8 /*lab_to_hid_syns*/;

	// Label input neurons to each respective pyramidal group (non plastic)
	this->ConnectNeurons(this->label_inputs[class_1], this->pyr_lists[class_1], 0, 0.5*n_neurons*0.8*lab_to_hid_syns, 1, -1, LabelWeight , true, false);
	this->ConnectNeurons(this->label_inputs[class_2], this->pyr_lists[class_2], 0, 0.5*n_neurons*0.8*lab_to_hid_syns, 1, -1, LabelWeight , true, false);
	//this->ConnectNeurons(this->label_inputs[class_3], this->pyr_lists[class_3], 0, 0.25*n_neurons*0.8*lab_to_hid_syns, 1, -1, LabelWeight , true, false);
	//this->ConnectNeurons(this->label_inputs[class_4], this->pyr_lists[class_4], 0, 0.25*n_neurons*0.8*lab_to_hid_syns, 1, -1, LabelWeight , true, false);
			
	//BINARY CLASSIFICATION EXCLUTIONS
	/* 
	this->ConnectNeurons(this->label_inputs_2, this->pyr_list_2, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_3, this->pyr_list_3, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_4, this->pyr_list_4, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_5, this->pyr_list_5, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_6, this->pyr_list_6, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_7, this->pyr_list_7, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_8, this->pyr_list_8, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	this->ConnectNeurons(this->label_inputs_9, this->pyr_list_9, 0, 0.1*baseSyns, 1, -1, LabelWeight , false);
	*/


	/*
	EXPERIMENTING WITH LATERAL INHIBITION PT2
	*/

	//PYR -> LATINH
	this->ConnectNeurons(this->pyr_lists[class_1], this->in_pv_0, 0, 0.5*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	this->ConnectNeurons(this->pyr_lists[class_2], this->in_pv_1, 0, 0.5*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->pyr_lists[class_3], this->in_pv_2, 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->pyr_lists[class_4], this->in_pv_3, 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);




	//LATINH -> PYR
	this->ConnectNeurons(this->in_pv_0, this->pyr_lists[class_2], 0, 0.5*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_0, this->pyr_lists[class_3], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_0, this->pyr_lists[class_4], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);





	this->ConnectNeurons(this->in_pv_1, this->pyr_lists[class_1], 0, 0.5*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_1, this->pyr_lists[class_3], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_1, this->pyr_lists[class_4], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);




	//this->ConnectNeurons(this->in_pv_2, this->pyr_lists[class_1], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_2, this->pyr_lists[class_2], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_2, this->pyr_lists[class_4], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);



	//this->ConnectNeurons(this->in_pv_3, this->pyr_lists[class_1], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_3, this->pyr_lists[class_2], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);
	//this->ConnectNeurons(this->in_pv_3, this->pyr_lists[class_3], 0, 0.25*n_neurons*0.8*lat_inh_syns, 1, -1, 1 , true, false);



}




// The main loop that simulates the stimulation dynamics, voltages,  Ca influx etc. 
// The time step is 1msec 
//
void LANetwork::StimDynamics(int duration)  // duration in msec
{
	int t = 0;
	bool spikeState[this->neurons.size()+1];
	int lastSpikeT[this->neurons.size()+1];


	// Zero out spike states
	fill_n(spikeState, this->neurons.size()+1, 0);
	fill_n(lastSpikeT, this->neurons.size()+1, 0);

	// zero out calcium
	for (syn_iter si = this->synapses.begin(); si != this->synapses.end(); ++si)
	{
		(*si)->calcium = 0.0;
	}

	// zero out neuron voltages
	for (nrn_iter ni = this->neurons.begin(); ni != this->neurons.end(); ++ni)
	{
		LANeuron* n = *ni;
		n->wadapt = 0.0;
		n->vspike =0.;
		n->vreset =0.;
		n->V =0.;
	}


	// zero out branch calcium counters + voltages
	for (branch_iter bi=this->branches.begin(); bi != this->branches.end(); ++bi)
	{
		(*bi)->totcalc = 0.0;
		(*bi)->depol = 0.0;
		(*bi)->dspike = 0.0;
		(*bi)->dspikeT = -1;
	}


	// Run simulation steps for each msec
	for (t=0; t < duration; t++)
	{
		for (nrn_iter ni = this->neurons.begin(); ni != this->neurons.end(); ++ni)
		{
			LANeuron* n = *ni;
			float soma_inh =0; // Total excitatory input  depolarization from dendrites that will be converted to input current in the soma
			float soma_exc =0; // Total inhibitory  ^


			// dive into all branches and calculate local depolarization / calcium etc
			for (branch_iter bi=n->branches.begin(); bi != n->branches.end(); ++bi)
			{
				LABranch* b = *bi;
				float dend_exc =0.;
				float dend_inh =0.;
				for (syn_iter si=b->synapses.begin(); si != b->synapses.end(); ++si)
				{
					LASynapse* s = *si;
					if (spikeState[s->source_nrn->nid])
					{
						if (s->source_nrn->type == 'V' ) dend_inh += ( s->weight);
						else if (s->source_nrn->type == 'M' ) soma_inh += ( s->weight);
						else dend_exc += (  s->weight);
					}
				}

				if (b->nlType == DEND_SUB)
				{
					// sublinear integration 
					b->depol +=  pow(4.0*dend_exc - 3.*dend_inh, 0.7) -  b->depol/20.0;
				}
				else if (b->nlType == DEND_SUPRA) 
				{
					// supralinear integration 
					b->depol +=  (4.0*dend_exc - 3. * dend_inh) -  b->depol/20.0; 
					if (b->dspikeT < t-70 && (n->vspike + b->depol) > this->dendSpikeThresh) // Generate a dendritic branch spike
					{
						b->depol = 50;
						b->dspikeT =t;
						b->branch_spikes++;
						n->dend_spikes++;
					}
				}
				else
				{
					// linear integration 
					b->depol +=  (4.0*dend_exc - 3.*dend_inh) -  b->depol/20.0;
				}

				// Sum up calcium influx from synapses
				for (syn_iter si=b->synapses.begin(); si != b->synapses.end(); ++si)
				{
					LASynapse* s = *si;
					if (spikeState[s->source_nrn->nid])
					{
						if (this->enablePlasticity && s->isPlastic) 
						{
							float depol =  b->depol + n->vspike;
							if (depol > 1.0)
							{
								// Simulate NMDA depolraization dependence
								//
								float ff =  (1.0/(1.+exp( (-(depol-30.0)/5.0))));

								s->calcium +=  1.1*ff; //NIKOS:IF I CHANGE SIMULATION TIME(duration) OR TIMESTEP, I CHANGE THIS AMOUNT ACCORDINGLY
							}
						}
					}
				}
				soma_exc +=  (b->depol); 
			}


			// Convert total depolarization to input currents for soma
			soma_exc *= 0.12;
			soma_inh *= 0.18;


			if (n->type == 'S') // Source/stimulation neuron
			{
				LAInput* in = (LAInput*)n;
				if (in->spikeTimes && t >= in->nextSpikeT && in->nextSpikeT >0)
				{
					// Time to emit a spike
					if (in->curSpike < in->totalSpikes)
						in->nextSpikeT = in->spikeTimes[in->curSpike++];
					else 
						in->nextSpikeT = -1;

					spikeState[in->nid] = 1;
					n->isSpiking = true;
					lastSpikeT[in->nid] = t;
					in->total_spikes++;
					in->V = 20.0; 
				}
				else
				{
					spikeState[in->nid] = 0;
					n->isSpiking = false;
					in->V = 0.0;
				}
			}
			else /// Pyr or interneuron
			{


				if (spikeState[n->nid])
				{
					// Just emitted a spike - reset voltage  and increase adaptation
					n->V = 0.0;
					n->wadapt += 0.18;
				}

				if (n->type == 'V') // PV interneuron
	 			{
					n->V +=  (soma_exc - soma_inh) - (n->V)/10.; 
				}
				else if (n->type == 'M') // SOM interneuron
	 			{
					n->V +=  (soma_exc - soma_inh) - (n->V)/10.; 
				}
				else
				{
					// Pyr neuron




					// Generate dCaAP  spike if needed
					if (this->hasCaap && n->type == 'P')
					{
						if (t - n->dCaapT  <  40)
						{
							//soma_exc = 1. + 4.5 *d/20.;
							//soma_exc = 1. + 3.5 *d/80.;
							//soma_exc =  soma_exc + 10. / ((float)(t - n->dCaapT)/1.);
							
							soma_exc =  soma_exc + (10. - soma_exc) / ((float)(t - n->dCaapT)/1.); // Linear model of dcap voltage
						} 
						else if (n->dCaapT < t - 200) 
						{
							if ( (n->Vfilt > 1.0 && n->Vfilt < 1.5) )  // if Vfilt  is between these thresholds, only  then elicit a dCaAP
							{
								n->dCaapT = t;
								n->totalCaap ++;
							}
						}

						n->Vfilt = (soma_exc - n->Vfilt ) / 20.;  // Vfilt is a low-pass filtered version of the total synaptic input current
					}


					n->V +=  soma_exc - 3.0*soma_inh - (n->V)/30. -   n->wadapt*(n->V+10.0) ; // Somatic unit = leaky integrate and fire  with adaptation


					// CREB activation changes the adaptation dynamics
					if (this->disableCreb)
						n->wadapt -= n->wadapt/180.;
					else
						n->wadapt -= n->wadapt/((180. - 70.0*(n->crebLevel>0.2 ? 1. : 0.)));
				}


				//Threshold for spike -- CREB also changes the threshold for spiking
				if ( lastSpikeT[n->nid] < t-2 && n->V > (20.0 - (n->crebLevel>100. ? 2.0 : 0.) ))
				{
					// Generate a somatic spike
					spikeState[n->nid] = 1;
					lastSpikeT[n->nid] = t;
					n->total_spikes++;
					n->isSpiking = true;
					n->vspike = 30.0;  // backpropagating spike voltage to add
					n->V = 70.;
				}
				else
				{
					if (n->isSpiking)
					{
						spikeState[n->nid] = 0;
						n->isSpiking = false;
					}
				}

				// backpropagating spike  decay
				if (n->vspike > 0) n->vspike -= n->vspike / 17.0; 

				// adaptation cap
				if (n->wadapt <-0.) n->wadapt =0.;

			}

			if (spikeState[n->nid])
			{
				// Record this spike(Could remove or remodel)
				n->spikeTimings.push_back(t+ this->Tstimulation);
			}
		}

	}


	this->Tstimulation += duration;
}




static void updminmax(float&  cmin, float& cmax, float val)
{
	if (val < cmin )
		cmin = val;
	if (val > cmax )
		cmax = val;
}



// Inter-stimulus dynamics: Protein synthesis , consolidation of synaptic tags to weights, tag decay, synaptic turnover etc
// Time step for simulation is 60 sec

void LANetwork::Interstim(int durationSecs)
{
	int tstop = T + durationSecs;
	this->isInterstim = true;

	printf("Interstim %d seconds (T=%d) plast=%d G=%d, L=%d ... \n", durationSecs, T, this->enablePlasticity, this->globalProteins, this->localProteins);

	float tstep = 60.0;
	int totalWeak =0;
	float weightLimit =  initWeight + 0.0;

	// Count the total weak afferent synapses
	for (input_iter ii = this->inputs_cs.begin(); ii != this->inputs_cs.end(); ++ii)
	{
		nrn_list lst = *ii;
		for (nrn_iter n = lst.begin(); n != lst.end(); ++n)
			for (syn_iter si=(*n)->outgoing.begin(); si != (*n)->outgoing.end(); ++si)
				if ((*si)->weight <= weightLimit)
					totalWeak++;
	}




	// bunch of counters for  stats / diagnostics
	int trec =0, tactrec=0;
	int totTagged=0, totTaggedMinus=0, totBProd =0;
	float totLTP=0., totLTD=0.;
	int totact =0, totSact =0;
	int totbspikes =0, totSpikes=0;
	float maxSpk =0;
	

	float actmin=9999, actmax=0; 
	float nactmin=9999, nactmax=0;




	for (nrn_iter ni = this->pyr_list.begin(); ni != this->pyr_list.end(); ni++)
	{
		LANeuron*  nrn = *ni;
		float nrnCalc =0.0;
		if (nrn->total_spikes > 4)
			totSact++;


		for (branch_iter bi = nrn->branches.begin(); bi != nrn->branches.end(); ++bi)
		{
			LABranch* b = *bi;
			totbspikes += b->branch_spikes;

			if (!this->enablePlasticity)
				continue;

			for (syn_iter si = b->synapses.begin(); si != b->synapses.end(); ++si)
			{
				LASynapse* s =*si;


				// Generate synaptic tags 
				float ctag = caDP(s->calcium);
				if (fabs(s->stag) < 0.1) /// Do not erase existing synaptic tags
				{
					s->stag = ctag;
					
					if (s->stag > 0.1)
					{
						totTagged++;
						totLTP += s->stag;
						//if (s->stag > 0.5 && b->prpTransients.size()>0) cout <<"Candidate is "<<s->sid<< " " << b->bid << " "<< nrn->nid<<endl;
					}
					if (s->stag < -0.1)
					{
						totTaggedMinus++;
						totLTD += s->stag;
					}	
				}


				b->totcalc += s->calcium;
				s->calcium = 0.;
			}


			//  if total calcium exceeds Branch protein synthesis threshold , then a new PRP concentration transient should start at this time T
			if ( b->totcalc  > this->localPRPThresh) // This branch should produce PRPs now BPROD
			{
				b->prpTransients.push_back( pair<float,float>(T, b->totcalc));
				totBProd++;
			}

			nrnCalc +=  b->totcalc;

		}


		if (nrn->total_spikes > CUTOFF*4.0) // Count active neurons
		{
			totSpikes += nrn->total_spikes;
			totact++;
			updminmax(actmin, actmax, nrnCalc);
		}
		else
			updminmax(nactmin, nactmax, nrnCalc);


		if (maxSpk < nrn->total_spikes)
			maxSpk = nrn->total_spikes;


		if (this->enablePlasticity)
		{

			if (nrnCalc > this->globalPRPThresh) // Global (somatic) protein synthesis threshol
			{
				nrn->prpTransients.push_back( pair<float,float>(T, nrnCalc) );

				if (!this->disableCreb ) 
					nrn->crebLevel=1.0;

				if (nrn->total_spikes > CUTOFF*4)
					tactrec ++;

				trec ++;
			}
		}



		nrn->totcalc  = nrnCalc;

	}

	//COMMENTED OUT BY NIKOS
	//printf("\n\nactive neuron spiked=[%f,%f] nonactive spikes=[%f,%f] \n", actmin, actmax, nactmin, nactmax);

	
	if (this->runningMode == RUN_TRAINING)
	{
		char buf[256];
		sprintf(buf,  "./data/r%d.dat", this->runningPatternNo);
		//this->SaveSnapshot(buf);
	}
	
	//COMMENTED OUT BY NIKOS
	//printf("Syn tags: +%d/-%d +%.1f/-%.1f G-PRPs:%d (%.1f%%) B-PRPs:%d, Act.:%d (%.1f%% ), Act+G-PRP:%d Avg Freq:%.1f max %.1f dSpikes:%d sact:%d\n", totTagged, totTaggedMinus, totLTP, totLTD, trec, 100.*(float)trec/(float(this->pyr_list.size())), totBProd, totact, 100.*(float)totact/(float(this->pyr_list.size())), tactrec, (float)totSpikes/((float)this->pyr_list.size()*4.0), float(maxSpk)/4.0, totbspikes, totSact);



	// Main loop for interstimulus dynamics 
	for (; T < tstop; T+= tstep)
	{
		for (nrn_iter ni = this->pyr_list.begin(); ni != this->pyr_list.end(); ni++)
		{
			LANeuron*  nrn = *ni;

			float totalSynapseWeight =0.0;
			float totalBranchStrength =0.0;
			int totalSynapses =0;
			nrn->protein =0.0;

			// "Whole-neuron" distribution of proteins
			nrn->proteinRate =0;
			for (pair_iter ii = nrn->prpTransients.begin(); ii != nrn->prpTransients.end(); ++ii)
			{
				pair<float, float> p = *ii;
				int td= (T - p.first);
				float al = (nuclearproteinalpha(td));
				if (nrn->proteinRate < al)
					nrn->proteinRate =  al;
			}

			
			
			for (branch_iter bi = nrn->branches.begin(); bi != nrn->branches.end(); ++bi)
			{
				LABranch* b = *bi;
				b->proteinRate =0.;

				for (pair_iter ii = b->prpTransients.begin();ii != b->prpTransients.end(); ++ii)
				{
					pair<float, float> p = *ii;
					float td = float(T - p.first);
					float al = (branchproteinalpha(td));
					if (b->proteinRate < al)
						b->proteinRate = al;
				}


				float  f =0.;

				if (this->localProteins)
					f = 1.0*b->proteinRate; 
				else if (this->globalProteins)
					f =  1.0* nrn->proteinRate;
				else
				{
					f = 1.0*b->proteinRate + 1.0* nrn->proteinRate;
					if (f>1.0) f = 1.0;
				}


				b->protein = f; 


				vector<LASynapse*> candidates;

				//NIKOS: HERE IS WHERE WEIGHTS ARE GETTING UPDATED. LET'S ADD A LEARNING RATE

				this->learning_rate = 1;

				for (syn_iter si = b->synapses.begin(); si != b->synapses.end(); ++si)
				{
					LASynapse* s =*si;

					if (s->stag != 0.0) 
					{
						s->stag -= (tstep/3600.)* s->stag;

						if (b->protein > 0.1 && (s->stag >0.1 || s->stag < 0.1))
						{
							//Convert  some of symaptic tag to synaptic weight
							
							float fw = s->stag* b->protein;
							//if (s->stag >0.)
							s->weight += s->syn_learning_rate * this->learning_rate * tstep * fw/400.; 
						}
					}

					if (s->weight > maxWeight)
						s->weight = maxWeight;
					else if (s->weight < 0.)
						s->weight = 0.;

					//***Nikos syn_learning_rate update addition here*****

					//***WITH IFS***
					/*
					if (s->weight >= 0.4 && s->weight < 0.6)
					{
						s->syn_learning_rate = 0.001;
					}
					*/


					/*
					else if (s->weight >= 0.6 && s->weight < 0.8)
					{
						s->syn_learning_rate = 0.0001;
					}
					else if (s->weight >= 0.8 && s->weight < 1.)
					{
						s->syn_learning_rate = 0.00001;
					}
					*/

					//***WITH SIGMOIDAL*** //slope = sigmoidal slope, midpos= position of "middle" sigmoidal point
					s->syn_learning_rate = -(this -> maxlr - this -> minlr ) /(1 + exp(-(this -> lr_slope) * (s->weight - this -> lr_midpos))) + this -> maxlr;


					//****************************************************
					totalSynapseWeight += s->weight;
					totalSynapses++;

					// Homeostasis
					// Synaptic scaling  of total synaptic weights
					s->weight += s->weight * (1.0 - nrn->synScaling )*tstep/(7.*24.0* 3600.*homeostasisTimeParam);


					/*if (this->debugMode && b->bid == DEBUG_BID)
					{
						cout << b->proteinRate << " " << nrn->proteinRate << " " << b->protein << " " << s->stag << " " << s->weight << " " << endl;
						this->mfile << " "  <<   s->stag << " " << s->weight ;
					}
					*/
				}

				/*
				if (T%800 ==0)
				{
					b->branchProteinHistory.push_back(b->protein);
				}
				*/

			}

			// Synaptic homeostasis / synaptic scaling
			if (totalSynapses>0)
				nrn->synScaling = totalSynapseWeight / (initWeight*float(totalSynapses));
		
			else
				nrn->synScaling = 1.0;

			// Branch plasticity homeostasis
			nrn->branch_scaling = totalBranchStrength/((float(1.0) * float(nrn->branches.size())));

			//CREB decay
			if (nrn->crebLevel >0.0)
			{
				nrn->crebLevel -= tstep/(3600.*8.*CREBTimeParam );
			}


		
		}


	}

	this->isInterstim = false;
}
