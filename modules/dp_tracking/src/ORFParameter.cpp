#include <iostream>
#include <assert.h>
#include "ORFInternal.h"

using namespace std;

/*******************************************************************************
	Constructor.
*******************************************************************************/
CORFParameters::CORFParameters() 
{
    // Forest
	numClasses = 2;
    numTrees = 10;
    numEpochs = 1;
    useSoftVoting = 1;
	pFeatureSpace = NULL;

	// Node/Tree
    maxDepth = 10;
    numRandomTests = 10;
	numFeatures = 10;
    numProjFeatures = 2;
    counterThreshold = 100;

    // Output
    verbose = 0;
}

/*******************************************************************************
	Load parameters from config file.
*******************************************************************************/
void CORFParameters::Load(FILE *pf)
{
	//printf("CORFParameters load\n");
	size_t rsize;
	int tag;
	rsize = fread(&tag, sizeof(int), 1, pf);
	if (rsize != 1)
		return;
	assert(tag = BE32('p', 'a', 'r', 'a'));
	rsize = fread(&numClasses, sizeof(int), 1, pf);
	rsize = fread(&numTrees, sizeof(int), 1, pf);
	rsize = fread(&numEpochs, sizeof(int), 1, pf);
	rsize = fread(&useSoftVoting, sizeof(int), 1, pf);
	rsize = fread(&maxDepth, sizeof(int), 1, pf);
	rsize = fread(&numFeatures, sizeof(int), 1, pf);
	rsize = fread(&numRandomTests, sizeof(int), 1, pf);
	rsize = fread(&numProjFeatures, sizeof(int), 1, pf);
	rsize = fread(&counterThreshold, sizeof(int), 1, pf);
	rsize = fread(&verbose, sizeof(int), 1, pf);
}

/*******************************************************************************
	Store parameters to config file.
*******************************************************************************/
void CORFParameters::Store(FILE *pf)
{
	int tag = BE32('p', 'a', 'r', 'a');
	fwrite(&tag, sizeof(int), 1, pf);
	fwrite(&numClasses, sizeof(int), 1, pf);
	fwrite(&numTrees, sizeof(int), 1, pf);
	fwrite(&numEpochs, sizeof(int), 1, pf);
	fwrite(&useSoftVoting, sizeof(int), 1, pf);
	fwrite(&maxDepth, sizeof(int), 1, pf);
	fwrite(&numFeatures, sizeof(int), 1, pf);
	fwrite(&numRandomTests, sizeof(int), 1, pf);
	fwrite(&numProjFeatures, sizeof(int), 1, pf);
	fwrite(&counterThreshold, sizeof(int), 1, pf);
	fwrite(&verbose, sizeof(int), 1, pf);
}

void CORFParameters::Print()
{
	printf("CORFParameters\n");
	printf("  numClasses = %d\n", numClasses);
	printf("  numTrees = %d\n", numTrees);
	printf("  numEpochs = %d\n", numEpochs);
	printf("  useSoftVoting = %d\n", useSoftVoting);
	printf("  maxDepth = %d\n", maxDepth);
	printf("  numFeatures = %d\n", numFeatures);
	printf("  numRandomTests = %d\n", numRandomTests);
	printf("  numProjFeatures = %d\n", numProjFeatures);
	printf("  counterThreshold = %d\n", counterThreshold);
	printf("  verbose = %d\n", verbose);
}
