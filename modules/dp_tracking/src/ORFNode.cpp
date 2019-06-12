#include <memory.h>
#include <assert.h>
#include "ORFInternal.h"

/*******************************************************************************
	Constructor for root node.
*******************************************************************************/
ORFNode::ORFNode(CORFParameters& par) 
	: m_pPar(&par), m_depth(0), m_isLeaf(true), m_counter(0.0), 
	m_parentCounter(0.0), m_label(-1)
{
	m_numClasses = m_pPar->numClasses;
	memset(m_labelStats, 0, sizeof(m_labelStats));

	// Creating random tests
	for (int i = 0; i < m_pPar->numRandomTests; i++) {
		RandomTest test(*m_pPar);
		m_tests.push_back(test);
	}
}

/*******************************************************************************
	Constructor for a child node.
*******************************************************************************/
ORFNode::ORFNode(ORFNode& parent) 
	: m_pPar(parent.m_pPar), m_isLeaf(true), m_counter(0.0), 
	m_parentCounter(0.0), m_label(-1)
{
	// parameters
	m_depth = parent.m_depth + 1;
	m_numClasses = m_pPar->numClasses;
	m_numFeatures = m_pPar->numFeatures;

	// copy parent's label statistics
	memcpy(m_labelStats, parent.m_labelStats, sizeof(m_labelStats));

	// compute label and update parent counter
	double max = -99999.0;
	for (int i = 0; i < m_numClasses; i++) {
		m_parentCounter += m_labelStats[i];
		if (m_labelStats[i] > max) {
			m_label = i;
			max = m_labelStats[i];
		}
	}

	// Creating random tests
	for (int i = 0; i < m_pPar->numRandomTests; i++) {
		RandomTest test(*m_pPar);
		m_tests.push_back(test);
	}
}

/*******************************************************************************
	Destructor.
*******************************************************************************/
ORFNode::~ORFNode() 
{
	if (!m_isLeaf) {
		delete m_leftChildNode;
		delete m_rightChildNode;
	}
}

/*******************************************************************************
	Evaluate a sample.
*******************************************************************************/
void ORFNode::Evaluate(Sample &sample, DecisionInfo& info) 
{
	// check if is leaf node
	if (m_isLeaf) {
		// check counter
		if (m_counter + m_parentCounter > 0.0) {
			// copy label statistics to result
			memcpy(info.result.confidence, m_labelStats, sizeof(m_labelStats));
			// normalize
			for (int i = 0; i < m_numClasses; i++) {
				info.result.confidence[i] *= 1.0 / (m_counter + m_parentCounter);
			}
			info.result.prediction = m_label;
		} else {
			// not enough info to make decision
			for (int i = 0; i < m_numClasses; i++) {
				info.result.confidence[i] = 1.0 / m_numClasses;
			}
			info.result.prediction = 0;
		}
		info.depth = m_depth;
		info.counter = m_counter;
	} else {
		// non-leaf node, choose a child node
		if (m_bestTest.Evaluate(sample)) {
			info.path |= 1 << m_depth;
			m_rightChildNode->Evaluate(sample, info);
		} else {
			m_leftChildNode->Evaluate(sample, info);
		}
	}
}

/*******************************************************************************
	Check if need to split.
*******************************************************************************/
bool ORFNode::ShouldISplit()
{
	bool isPure = false;
	for (int i = 0; i < m_numClasses; i++) {
		if (m_labelStats[i] == m_counter + m_parentCounter) {
			isPure = true;
			break;
		}
	}
	if (isPure) {
		return false;
	}
	// do not split if max depth is reached
	if (m_depth >= m_pPar->maxDepth) { 
		return false;
	}
	// do not split if not enough samples seen
	if (m_counter < m_pPar->counterThreshold) { 
		return false;
	}
	return true;
}

/*******************************************************************************
	Update with a sample for training.
*******************************************************************************/
void ORFNode::Update(Sample &sample) 
{
    m_counter += sample.w;
    m_labelStats[sample.y] += sample.w;

    if (m_isLeaf) {
        // Update online tests
        for (int i = 0; i < m_pPar->numRandomTests; i++) {
            m_tests[i].Update(sample);
        }

        // Update the label
        m_label = Argmax(m_labelStats, m_numClasses);

        // Decide for split
        if (ShouldISplit()) {
            m_isLeaf = false;

            // Find the best random test
            int maxIndex = 0;
            double maxScore = -1e10, score;
            for (int i = 0; i < m_pPar->numRandomTests; i++) {
                score = m_tests[i].ComputeScore();
                if (score > maxScore) {
                    maxScore = score;
                    maxIndex = i;
                }
            }
            m_bestTest = m_tests[maxIndex];
            m_tests.clear();

            // Split
            pair<vector<double>, vector<double> > parentStats = m_bestTest.GetStats();
            m_rightChildNode = new ORFNode(*this);
            m_leftChildNode = new ORFNode(*this);
        }
    } else { 
		// non-leaf node, continue evaluation with child node
        if (m_bestTest.Evaluate(sample)) {
            m_rightChildNode->Update(sample);
        } else {
            m_leftChildNode->Update(sample);
        }
    }
}

/*******************************************************************************
	Load from file.
*******************************************************************************/
void ORFNode::Load(FILE *pf)
{
	size_t rsize;
	int tag;
	rsize = fread(&tag, sizeof(int), 1, pf);
	if (rsize != 1)
		return;
	assert(tag == (int)BE32('n', 'o', 'd', 'e'));
	// is leaf
	rsize = fread(&m_isLeaf, sizeof(int), 1, pf);
	// counter
	rsize = fread(&m_counter, sizeof(double), 1, pf);
	// parent counter
	rsize = fread(&m_parentCounter, sizeof(double), 1, pf);
	// label statistics
	int size;
	rsize = fread(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp;
		rsize = fread(&temp, sizeof(double), 1, pf);
		m_labelStats[i] = temp;
	}

	if (m_isLeaf) {
		// online tests
		rsize = fread(&size, sizeof(int), 1, pf);
		m_tests.clear();
		for (int i = 0; i < size; i++) {
			RandomTest test(*m_pPar);
			test.Load(pf);
			m_tests.push_back(test);
		}
	} else {
		// best test
		m_bestTest.Load(pf);
	}

	// child nodes
	if (!m_isLeaf) {
		pair<vector<double>, vector<double> > parentStats = m_bestTest.GetStats();
		m_rightChildNode = new ORFNode(*this);
		m_leftChildNode = new ORFNode(*this);
		m_leftChildNode->Load(pf);
		m_rightChildNode->Load(pf);
	}
}

/*******************************************************************************
	Store into file.
*******************************************************************************/
void ORFNode::Store(FILE *pf)
{
	int tag = BE32('n', 'o', 'd', 'e');
	fwrite(&tag, sizeof(int), 1, pf);
	// is leaf
	fwrite(&m_isLeaf, sizeof(int), 1, pf);
	// counter
	fwrite(&m_counter, sizeof(double), 1, pf);
	// parent counter
	fwrite(&m_parentCounter, sizeof(double), 1, pf);
	// label statistics
	int size = m_pPar->numClasses;
	fwrite(&size, sizeof(int), 1, pf);
	for (int i = 0; i < size; i++) {
		double temp = m_labelStats[i];
		fwrite(&temp, sizeof(double), 1, pf);
	}

	if (m_isLeaf) {
		// online tests
		size = m_tests.size();
		fwrite(&size, sizeof(int), 1, pf);
		for (int i = 0; i < size; i++) {
			m_tests[i].Store(pf);
		}
	} else {
		// best test
		m_bestTest.Store(pf);
	}

	// child nodes
	if (!m_isLeaf) {
		m_leftChildNode->Store(pf);
		m_rightChildNode->Store(pf);
	}
}

/*******************************************************************************
	Analyse node.
*******************************************************************************/
void ORFNode::Analyse()
{
	m_pPar->stats.iTotalNodes++;

	if (!m_isLeaf) {
		m_leftChildNode->Analyse();
		m_rightChildNode->Analyse();
		return;
	}

	m_pPar->stats.iTotalLeafNodes++;
}
