#ifndef _ORFINTERNAL_H_
#define _ORFINTERNAL_H_

#include <cstdlib>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <assert.h>
#ifndef WIN32
#include <sys/time.h>
#endif

#include "ORF.h"

using namespace std;

#define BE32(a, b, c, d)	(((d) << 24) + ((c) << 16) + ((b) << 8) + (a))

// utility functions
unsigned int GetDevRandom();
void RandPerm(const int &inNum, vector<int> &outVect);
void RandPerm(const int &inNum, const int inPart, vector<int> &outVect);

/*******************************************************************************
	Decision path info.
*******************************************************************************/
typedef struct  
{
	Result result;			// result
	unsigned int path;		// decision path
	int depth;				// decision leaf depth
	double counter;			// decision leaf counter
} DecisionInfo;

/*******************************************************************************
	Random test class.
*******************************************************************************/
class RandomTest
{
public:
	RandomTest() {}
	RandomTest(CORFParameters &par);

	void UpdateStats(const Sample &sample, const bool decision);
	double ComputeScore();
	pair<vector<double>, vector<double> > GetStats();
	void Update(Sample &sample);
	bool Evaluate(Sample &sample);
	void Load(FILE *pf);
	void Store(FILE *pf);

private:
	int		m_numClasses;			// number of classes
	double	m_threshold;			// threshold
	double	m_trueCount;			// true count
	double	m_falseCount;			// false count
	int		m_numProjFeatures;		// number of projection features
	vector<double>	m_trueStats;	// true statistics
	vector<double>	m_falseStats;	// false statistics
	vector<int>		m_features;		// randomized feature index list
	vector<double>	m_weights;		// randomized feature weights
};

/*******************************************************************************
	ORF node class.
*******************************************************************************/
class ORFNode 
{
public:
	ORFNode(CORFParameters &par);
	ORFNode(ORFNode& parent);
	~ORFNode();

	void Update(Sample &sample);
	void Evaluate(Sample &sample, DecisionInfo& info);

	void Load(FILE *pf);
	void Store(FILE *pf);

	void Analyse();

private:
	CORFParameters* m_pPar;			// parameters
	int     m_numClasses;			// number of classes
	int     m_numFeatures;			// number of features to use
	int     m_depth;				// current node depth
	int     m_isLeaf;				// if a leaf node
	double  m_counter;				// counter of sample weights
	double  m_parentCounter;		// parent counter
	int     m_label;				// node label
	double  m_labelStats[MAX_NUM_CLASSES];	// label statistics

	ORFNode* m_leftChildNode;
	ORFNode* m_rightChildNode;

	vector<RandomTest> m_tests;		// list of online tests
	RandomTest m_bestTest;			// best test for this node

	// check if should split the node
	bool ShouldISplit();
};

/*******************************************************************************
	ORF tree class.
*******************************************************************************/
class ORFTree
{
public:
	ORFTree(CORFParameters& par);
	~ORFTree();

	void Update(Sample &sample);
	void Train(SampleSet &dataset);
	void Evaluate(Sample &sample, Result& result);

	vector<Result> Test(SampleSet &dataset);
	vector<Result> TrainAndTest(SampleSet &dataset_tr, SampleSet &dataset_ts);

	void Load(FILE *pf);
	void Store(FILE *pf);

	double GetOOBE();
	void UpdateOOBE(int decision);
	void Analyse();

private:
	CORFParameters* m_pPar;			// parameters
	ORFNode*		m_rootNode;		// tree nodes
	double			m_counter;		// counter of sample weights
	double			m_oobe;			// oobe
	double			m_evalCount;    // evaluation count
	DecisionInfo	m_decision;		// decision info
};

/*******************************************************************************
	Inline functions.
*******************************************************************************/

/*******************************************************************************
	Return a random number in range [0.0, 1.0].
*******************************************************************************/
inline double RandDouble() 
{
	static bool didSeeding = false;

	if (!didSeeding) {
		unsigned int seedNum;
		seedNum = (unsigned int) time(NULL);
		srand(seedNum);
		didSeeding = true;
	}
	return rand() / (RAND_MAX + 1.0);
}

/*******************************************************************************
	Return a random number in range [min, max].
*******************************************************************************/
inline double RandomFromRange(const double &minRange, const double &maxRange) 
{
	return minRange + (maxRange - minRange) * RandDouble();
}

/*******************************************************************************
	Fill vector with random numbers with range [-1.0, +1.0]
*******************************************************************************/
inline void FillWithRandomNumbers(const int &length, vector<double> &inVect) 
{
	inVect.clear();
	for (int i = 0; i < length; i++) {
		inVect.push_back(2.0 * (RandDouble() - 0.5));
	}
}

/*******************************************************************************
	Compute argmax
*******************************************************************************/
inline int Argmax(double* vect, int size) 
{
	double maxValue = vect[0];
	int maxIndex = 0;
	for (int i = 1; i < size; i++) {
		if (vect[i] > maxValue) {
			maxValue = vect[i];
			maxIndex = i;
		}
	}

	return maxIndex;
}

/*******************************************************************************
	Compute sum of vector.
*******************************************************************************/
inline double Sum(const vector<double> &inVect) 
{
	double val = 0.0;
	vector<double>::const_iterator itr(inVect.begin()), end(inVect.end());
	while (itr != end) {
		val += *itr;
		++itr;
	}

	return val;
}

/*******************************************************************************
	Poisson sampling.
*******************************************************************************/
inline int Poisson(double A) 
{
	int k = 0;
	int maxK = 10;
	while (1) {
		double U_k = RandDouble();
		A *= U_k;
		if (k > maxK || A < exp(-1.0)) {
			break;
		}
		k++;
	}
	return k;
}

#endif /* _ORFINTERNAL_H_ */
