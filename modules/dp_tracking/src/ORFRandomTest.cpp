#include <math.h>
#include "ORFInternal.h"

static double rf_log2(double x)
{
	return log(x) / log(2.0);
}

/*******************************************************************************
	Constructor.
*******************************************************************************/
RandomTest::RandomTest(CORFParameters &hp) 
	: m_trueCount(0.0), m_falseCount(0.0) 
{
	int i;

	m_numClasses = hp.numClasses;
	for (i = 0; i < m_numClasses; i++) {
		m_trueStats.push_back(0.0);
		m_falseStats.push_back(0.0);
	}
	m_numProjFeatures = hp.numProjFeatures;
	// randomize feature list
	RandPerm(hp.numFeatures, m_numProjFeatures, m_features);
	// randomize feature weights with range [-1.0, +1.0]
	FillWithRandomNumbers(m_numProjFeatures, m_weights);
	// create random threshold based on sum of weights
	double w = 0.0;
	for (i = 0; i < m_numProjFeatures; i++) {
		w += m_weights[i];
	}
	// FIXME: try different range
	m_threshold = RandomFromRange(0, 512) * w;
}

/*******************************************************************************
	Update statistics.
*******************************************************************************/
void RandomTest::UpdateStats(const Sample &sample, const bool decision) 
{
	if (decision) {
		m_trueCount += sample.w;
		m_trueStats[sample.y] += sample.w;
	} else {
		m_falseCount += sample.w;
		m_falseStats[sample.y] += sample.w;
	}
}

/*******************************************************************************
	Compute score.
*******************************************************************************/
double RandomTest::ComputeScore() 
{
	double totalCount = m_trueCount + m_falseCount;

	// Split Entropy
	double p, splitEntropy = 0.0;
	if (m_trueCount) {
		p = m_trueCount / totalCount;
		splitEntropy -= p * rf_log2(p);
	}
	if (m_falseCount) {
		p = m_falseCount / totalCount;
		splitEntropy -= p * rf_log2(p);
	}

	// Prior Entropy
	double priorEntropy = 0.0;
	for (int i = 0; i < m_numClasses; i++) {
		p = (m_trueStats[i] + m_falseStats[i]) / totalCount;
		if (p) {
			priorEntropy -= p * rf_log2(p);
		}
	}

	// Posterior Entropy
	double trueScore = 0.0, falseScore = 0.0;
	if (m_trueCount) {
		for (int i = 0; i < m_numClasses; i++) {
			p = m_trueStats[i] / m_trueCount;
			if (p) {
				trueScore -= p * rf_log2(p);
			}
		}
	}
	if (m_falseCount) {
		for (int i = 0; i < m_numClasses; i++) {
			p = m_falseStats[i] / m_falseCount;
			if (p) {
				falseScore -= p * rf_log2(p);
			}
		}
	}
	double posteriorEntropy = (m_trueCount * trueScore + m_falseCount * falseScore) / totalCount;

	// Information Gain
	return (2.0 * (priorEntropy - posteriorEntropy)) / (priorEntropy * splitEntropy + 1e-10);
}

/*******************************************************************************
	Get statistics.
*******************************************************************************/
pair<vector<double>, vector<double> > RandomTest::GetStats() 
{
	return pair<vector<double>, vector<double> > (m_trueStats, m_falseStats);
}

/*******************************************************************************
	Update sample statistics
*******************************************************************************/
void RandomTest::Update(Sample &sample) 
{
	UpdateStats(sample, Evaluate(sample));
}

/*******************************************************************************
	Evaluate a sample.
*******************************************************************************/
bool RandomTest::Evaluate(Sample &sample) 
{
	double proj = 0.0;
	for (int i = 0; i < m_numProjFeatures; i++) {
		proj += sample.x[m_features[i]] * m_weights[i];
	}
	return (proj > m_threshold) ? true : false;
}

/*******************************************************************************
	Load from file.
*******************************************************************************/
void RandomTest::Load(FILE *pf)
{
	int tag;
	size_t rsize;
	rsize = fread(&tag, sizeof(int), 1, pf);
	if (rsize != 1)
		return;
	assert(tag == BE32('t', 'e', 's', 't'));
	// number of classes
	rsize = fread(&m_numClasses, sizeof(int), 1, pf);
	// threshold
	rsize = fread(&m_threshold, sizeof(double), 1, pf);
	// true count
	rsize = fread(&m_trueCount, sizeof(double), 1, pf);
	// false count
	rsize = fread(&m_falseCount, sizeof(double), 1, pf);
	// true statistics
	int size;
	rsize = fread(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp;
		rsize = fread(&temp, sizeof(double), 1, pf);
		if ((int)m_trueStats.size() > i) {
			m_trueStats[i] = temp;
		} else {
			m_trueStats.push_back(temp);
		}
	}
	// false statistics
	rsize = fread(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp;
		rsize = fread(&temp, sizeof(double), 1, pf);
		if ((int)m_falseStats.size() > i) {
			m_falseStats[i] = temp;
		} else {
			m_falseStats.push_back(temp);
		}
	}
	// number of projected features
	rsize = fread(&m_numProjFeatures, sizeof(int), 1, pf);
	// features
	rsize = fread(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		int temp;
		rsize = fread(&temp, sizeof(int), 1, pf);
		if ((int)m_features.size() > i) {
			m_features[i] = temp;
		} else {
			m_features.push_back(temp);
		}
	}
	// feature weights
	rsize = fread(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp;
		rsize = fread(&temp, sizeof(double), 1, pf);
		if ((int)m_weights.size() > i) {
			m_weights[i] = temp;
		} else {
			m_weights.push_back(temp);
		}
	}
}

/*******************************************************************************
	Store to file.
*******************************************************************************/
void RandomTest::Store(FILE *pf)
{
	int tag = BE32('t', 'e', 's', 't');
	fwrite(&tag, sizeof(int), 1, pf);
	// number of classes
	fwrite(&m_numClasses, sizeof(int), 1, pf);
	// threshold
	fwrite(&m_threshold, sizeof(double), 1, pf);
	// true count
	fwrite(&m_trueCount, sizeof(double), 1, pf);
	// false count
	fwrite(&m_falseCount, sizeof(double), 1, pf);
	// true statistics
	int size = m_trueStats.size();
	fwrite(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp = m_trueStats[i];
		fwrite(&temp, sizeof(double), 1, pf);
	}
	// false statistics
	size = m_falseStats.size();
	fwrite(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp = m_falseStats[i];
		fwrite(&temp, sizeof(double), 1, pf);
	}
	// number of projected features
	fwrite(&m_numProjFeatures, sizeof(int), 1, pf);
	// features
	size = m_features.size();
	fwrite(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		int temp = m_features[i];
		fwrite(&temp, sizeof(int), 1, pf);
	}
	// feature weights
	size = m_weights.size();
	fwrite(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp = m_weights[i];
		fwrite(&temp, sizeof(double), 1, pf);
	}
}
