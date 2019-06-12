#ifndef _ORF_H_
#define _ORF_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "FeatureSpace.h"

using namespace std;

const int MAX_NUM_CLASSES = 2;

/*******************************************************************************
	ORF statistics.
*******************************************************************************/
typedef struct {
	int	iTotalTrainings;
	int iTotalEvaluations;
	int iTotalNodes;
	int iTotalLeafNodes;
	int iEffectiveNodes;
} ORFStats;

/*******************************************************************************
	ORF parameters.
*******************************************************************************/
class CORFParameters
{
public:
	CORFParameters();
	void Load(FILE *pf);
	void Store(FILE *pf);
	void Print();

	int numClasses;			// number of classes
	int numTrees;			// number of trees
	int maxDepth;			// max tree depth
	int useSoftVoting;		// if use soft voting
	int numEpochs;			// training epochs
	int numFeatures;		// number of features
	// per node
	int numRandomTests;		// number of random tests
	int numProjFeatures;	// number of projection features
	int counterThreshold;	// counter threshold

	CFeatureSpace* pFeatureSpace;	// pointer of feature space
	int verbose;			// verbose control

	ORFStats stats;			// statistics
};

/*******************************************************************************
	Result class.
*******************************************************************************/
typedef struct Result 
{
    int prediction;						// final prediction
    double confidence[MAX_NUM_CLASSES];	// confidence of each class
} Result;

class ORFTree;

/*******************************************************************************
	Online random forest class.
*******************************************************************************/
class OnlineRF
{
public:
	// constructor
	OnlineRF(CORFParameters &hp);
	~OnlineRF();

	void Update(Sample &sample);
	void Train(SampleSet &dataset);
	void Evaluate(Sample &sample, Result& result);

	vector<Result> Test(SampleSet &dataset);
	vector<Result> TrainAndTest(SampleSet &dataset_tr, SampleSet &dataset_ts);

	void Load(FILE *pf);
	void Store(FILE *pf);

	void Analyse();
	int GetMaxTreeOOBE(double& oobe);
	void Prune(double thr);
	void DeleteTree(int treeId);

protected:
	int     m_numClasses;       // number of classes to be classified
	double  m_counter;          // counter of sample weights
	double  m_oobe;             // out-of-bag error
	CORFParameters *m_pPar;     // parameters
	vector<ORFTree*> m_trees;   // trees
};

#endif /* _ORF_H_ */
