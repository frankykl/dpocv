#include "ORFInternal.h"

/*******************************************************************************
	Get dev/random.
*******************************************************************************/
unsigned int GetDevRandom() 
{
    ifstream devFile("/dev/urandom", ios::binary);
    unsigned int outInt = 0;
    char tempChar[sizeof(outInt)];

    devFile.read(tempChar, sizeof(outInt));
    outInt = atoi(tempChar);

    devFile.close();

    return outInt;
}

/*******************************************************************************
	Random permutation.
*******************************************************************************/
void RandPerm(const int &inNum, vector<int> &outVect) 
{
    outVect.resize(inNum);
    int randIndex, tempIndex;
    for (int i = 0; i < inNum; i++) {
        outVect[i] = i;
    }
    for (register int nFeat = 0; nFeat < inNum; nFeat++) {
        randIndex = (int) floor(((double) inNum - nFeat) * RandDouble()) + nFeat;
        if (randIndex == inNum) {
            randIndex--;
        }
        tempIndex = outVect[nFeat];
        outVect[nFeat] = outVect[randIndex];
        outVect[randIndex] = tempIndex;
    }
}

/*******************************************************************************
	Random permutation.
	inNum - Total number.
	inPart - Number of parts.
	outVect - Output vector.
*******************************************************************************/
void RandPerm(const int &inNum, const int inPart, vector<int> &outVect) 
{
    outVect.resize(inNum);
    int randIndex, tempIndex;
    for (int i = 0; i < inNum; i++) {
        outVect[i] = i;
    }
    for (register int nFeat = 0; nFeat < inPart; nFeat++) {
        randIndex = (int) floor(((double) inNum - nFeat) * RandDouble()) + nFeat;
        if (randIndex == inNum) {
            randIndex--;
        }
        tempIndex = outVect[nFeat];
        outVect[nFeat] = outVect[randIndex];
        outVect[randIndex] = tempIndex;
    }

    outVect.erase(outVect.begin() + inPart, outVect.end());
}
