#include <memory.h>
#include "ORF.h"
#include "ORFInternal.h"

/*******************************************************************************
	Constructor.
*******************************************************************************/
OnlineRF::OnlineRF(CORFParameters &par) 
	: m_numClasses(par.numClasses), m_counter(0.0), m_oobe(0.0), m_pPar(&par)
{
	// Create trees
	for (int i = 0; i < par.numTrees; i++) {
		ORFTree *tree = new ORFTree(par);
		m_trees.push_back(tree);
	}
}

/*******************************************************************************
	Destructor.
*******************************************************************************/
OnlineRF::~OnlineRF() 
{
	// Delete trees
	for (int i = 0; i < m_pPar->numTrees; i++) {
		if (m_trees[i] != NULL)
			delete m_trees[i];
	}
}

/*******************************************************************************
	Update ORF with one single sample during training.
*******************************************************************************/
void OnlineRF::Update(Sample &sample) 
{
	m_counter += sample.w;

	// initialize result for each class
	Result result, treeResult;
	for (int i = 0; i < m_numClasses; i++) {
		result.confidence[i] = 0.0;
	}

	// compute sample features
	if (m_pPar->pFeatureSpace != NULL) {
		m_pPar->pFeatureSpace->ComputeSampleFeatures(sample);
	}

	// loop for all trees
	for (int i = 0; i < m_pPar->numTrees; i++) {
		// randomize training and evaluation based on Poisson distribution
		int numTries = Poisson(1.0);
		if (numTries) {
			for (int n = 0; n < numTries; n++) {
				m_trees[i]->Update(sample);
			}
		} else {
			m_trees[i]->Evaluate(sample, treeResult);
			if (m_pPar->useSoftVoting) {
				for (int j = 0; j < m_numClasses; j++) {
					result.confidence[j] += treeResult.confidence[j];
				}
			} else {
				result.confidence[treeResult.prediction]++;
			}

			// update OOBE
			if (Argmax(result.confidence, m_numClasses) != sample.y) {
				m_oobe += sample.w;
			}
		}
	}
}

/*******************************************************************************
	Train with data set.
*******************************************************************************/
void OnlineRF::Train(SampleSet &dataset) 
{
	vector<int> randIndex;
	for (int n = 0; n < m_pPar->numEpochs; n++) {
		//printf("epoch %d\n", n);
		// randomize samples
		RandPerm(dataset.size(), randIndex);
		// train with randomized samples
		for (int i = 0; i < (int)dataset.size(); i++) {
			//printf("sample %d, %d\n", i, randIndex[i]);
			Update(*(dataset[randIndex[i]]));
		}
	}
}

/*******************************************************************************
	Evaluate a sample.
*******************************************************************************/
void OnlineRF::Evaluate(Sample &sample, Result& result) 
{
	int i, j;

	// initialize result for each class
	Result treeResult;
	for (i = 0; i < m_numClasses; i++) {
		result.confidence[i] = 0.0;
	}

	// compute sample features
	if (m_pPar->pFeatureSpace != NULL) {
		m_pPar->pFeatureSpace->ComputeSampleFeatures(sample);
	}

	// evaluate sample with every tree
	for (i = 0; i < m_pPar->numTrees; i++) {
		m_trees[i]->Evaluate(sample, treeResult);
		//printf("  tree %d, %f, %f\n", i, treeResult.confidence[0], treeResult.confidence[1]);
		if (m_pPar->useSoftVoting) {
			for (j = 0; j < m_numClasses; j++) {
				result.confidence[j] += treeResult.confidence[j];
			}
		} else {
			result.confidence[treeResult.prediction]++;
		}
	}

	// scale and return result
	for (i = 0; i < m_numClasses; i++) {
		result.confidence[i] *= 1.0 / m_pPar->numTrees;
	}
	result.prediction = Argmax(result.confidence, m_numClasses);

	// update tree oobe
	for (i = 0; i < m_pPar->numTrees; i++) {
		m_trees[i]->UpdateOOBE(result.prediction);
	}
}

/*******************************************************************************
	Test with data set.
*******************************************************************************/
vector<Result> OnlineRF::Test(SampleSet &dataset) 
{
	vector<Result> results;
	for (int i = 0; i < (int)dataset.size(); i++) {
		Result r;
		Evaluate(*(dataset[i]), r);
		results.push_back(r);
	}

	return results;
}

/*******************************************************************************
	Train and test.
*******************************************************************************/
vector<Result> OnlineRF::TrainAndTest(SampleSet &dataset_tr, SampleSet &dataset_ts) 
{
	// train
	vector<Result> results;
	vector<int> randIndex;
	for (int n = 0; n < m_pPar->numEpochs; n++) {
		RandPerm(dataset_tr.size(), randIndex);
		for (int i = 0; i < (int)dataset_tr.size(); i++) {
			Update(*(dataset_tr[randIndex[i]]));
		}

		// run test
		results = Test(dataset_ts);
	}

	return results;
}

void OnlineRF::DeleteTree(int treeId)
{
	if (treeId >= m_trees.size())
		return;

	if (m_trees[treeId] != NULL) {
		delete m_trees[treeId];
		m_trees.erase(m_trees.begin() + treeId);
	}
	m_pPar->numTrees--;
}


/*******************************************************************************
	Load from file.
*******************************************************************************/
void OnlineRF::Load(FILE *pf) 
{
	size_t rsize;
	int tag;
	rsize = fread(&tag, sizeof(int), 1, pf);
	if (rsize != 1)
		return;
	assert(tag == BE32('f', 'r', 's', 't'));
	rsize = fread(&m_counter, sizeof(double), 1, pf);
	rsize = fread(&m_oobe, sizeof(double), 1, pf);
	for (int i = 0; i < m_pPar->numTrees; i++) {
		m_trees[i]->Load(pf);
	}
}

/*******************************************************************************
	Store to a file.
*******************************************************************************/
void OnlineRF::Store(FILE *pf) 
{
	int tag = BE32('f', 'r', 's', 't');
	fwrite(&tag, sizeof(int), 1, pf);
	fwrite(&m_counter, sizeof(double), 1, pf);
	fwrite(&m_oobe, sizeof(double), 1, pf);
	for (int i = 0; i < m_pPar->numTrees; i++) {
		m_trees[i]->Store(pf);
	}
}

/*******************************************************************************
	Analyse random forest.
*******************************************************************************/
void OnlineRF::Analyse()
{
	int i;

	memset(&m_pPar->stats, 0, sizeof(ORFStats));
	for (i = 0; i < m_pPar->numTrees; i++) {
		m_trees[i]->Analyse();
	}

	printf("\nRF statistics\n");
	printf("Total nodes = %d\n", m_pPar->stats.iTotalNodes);
	printf("Total leaf nodes = %d\n", m_pPar->stats.iTotalLeafNodes);
}

/*******************************************************************************
	Analyse random forest.
*******************************************************************************/
int OnlineRF::GetMaxTreeOOBE(double& oobe)
{
	int i, max_index = -1;
	double max = 0.0;
	printf("\nTree OOBE\n");
	for (i = 0; i < m_pPar->numTrees; i++) {
		double oobe = m_trees[i]->GetOOBE();
		printf("Tree %d = %4.2f\n", i, oobe);
		if (oobe > max) {
			max = oobe;
			max_index = i;
		}
	}

	oobe = max;
	return max_index;
}

void OnlineRF::Prune(double thr)
{
	double max;
	int max_index = GetMaxTreeOOBE(max);

	// Remove the tree with the biggest OOBE and grow a new one
	printf("max_index = %d, %f\n", max_index, max);
	if (max_index >= 0 && max > (double)thr) {
		printf("Remove tree %d\n", max_index);
		delete m_trees[max_index];
		m_trees.erase(m_trees.begin() + max_index);
		ORFTree *tree = new ORFTree(*m_pPar);
		m_trees.push_back(tree);
	}
}
