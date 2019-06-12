#include <memory.h>
#include "ORFInternal.h"

/*******************************************************************************
	Constructor.
*******************************************************************************/
ORFTree::ORFTree(CORFParameters &hp) 
	: m_pPar(&hp), m_counter(0.0), m_oobe(0.0), m_evalCount(0.0)
{
	// create root node
	m_rootNode = new ORFNode(hp);
}

/*******************************************************************************
	Destructor.
*******************************************************************************/
ORFTree::~ORFTree() {
	if (m_rootNode != NULL)
		delete m_rootNode;
}

/*******************************************************************************
	Update with a sample.
*******************************************************************************/
void ORFTree::Update(Sample &sample) 
{
	m_rootNode->Update(sample);
}

/*******************************************************************************
	Train with sample set.
*******************************************************************************/
void ORFTree::Train(SampleSet &dataset) 
{
	vector<int> randIndex;
	for (int n = 0; n < m_pPar->numEpochs; n++) {
		// random permutation of sample indices
		RandPerm(dataset.size(), randIndex);
		for (unsigned int i = 0; i < dataset.size(); i++) {
			Update(*(dataset[randIndex[i]]));
		}
	}
}

/*******************************************************************************
	Evaluate a sample.
*******************************************************************************/
void ORFTree::Evaluate(Sample &sample, Result& result) 
{
	memset(&m_decision, 0, sizeof(DecisionInfo));
	m_rootNode->Evaluate(sample, m_decision);
	memcpy(&result, &m_decision.result, sizeof(Result));
}

/*******************************************************************************
	Test with sample set.
*******************************************************************************/
vector<Result> ORFTree::Test(SampleSet &dataset) 
{
	vector<Result> results;
	for (unsigned int i = 0; i < dataset.size(); i++) {
		Result r;
		Evaluate(*(dataset[i]), r);
		results.push_back(r);
	}

	return results;
}

/*******************************************************************************
	Train and test.
*******************************************************************************/
vector<Result> ORFTree::TrainAndTest(SampleSet &dataset_tr, SampleSet &dataset_ts) 
{
	vector<Result> results;
	vector<int> randIndex;
	for (int n = 0; n < m_pPar->numEpochs; n++) {
		RandPerm(dataset_tr.size(), randIndex);
		for (unsigned int i = 0; i < dataset_tr.size(); i++) {
			Update(*(dataset_tr[randIndex[i]]));
		}

		results = Test(dataset_ts);
	}

	return results;
}

/*******************************************************************************
	Load from file.
*******************************************************************************/
void ORFTree::Load(FILE *pf) 
{
	size_t rsize;
	int tag;
	rsize = fread(&tag, sizeof(int), 1, pf);
	if (rsize != 1)
		return;
	assert(tag == BE32('t', 'r', 'e', 'e'));
	rsize = fread(&m_counter, sizeof(double), 1, pf);
	m_rootNode->Load(pf);
}

/*******************************************************************************
	Store to file.
*******************************************************************************/
void ORFTree::Store(FILE *pf) 
{
	int tag = BE32('t', 'r', 'e', 'e');
	fwrite(&tag, sizeof(int), 1, pf);
	fwrite(&m_counter, sizeof(double), 1, pf);
	m_rootNode->Store(pf);
}

/*******************************************************************************
	Update tree oobe.
*******************************************************************************/
void ORFTree::UpdateOOBE(int decision)
{
	m_evalCount += 1.0;
	if (m_decision.result.prediction == decision) {
		return;
	}

	if (m_decision.depth == m_pPar->maxDepth && m_decision.counter > 100.0) {
		m_oobe += 1.0;
	}
}

/*******************************************************************************
	Analyse tree.
*******************************************************************************/
void ORFTree::Analyse()
{
	m_rootNode->Analyse();
}

double ORFTree::GetOOBE() 
{
	double oobe = m_oobe / m_evalCount;
	return oobe;
}
