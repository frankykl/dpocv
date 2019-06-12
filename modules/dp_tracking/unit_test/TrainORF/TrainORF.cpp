#include <iostream>
#include <assert.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../../src/FeatureSpace.h"
#include "../../src/ORF.h"
#include "../../src/ORFInternal.h"

using namespace cv;

// Training parameter strucutre
typedef struct TrainingParam {
public:
	// constructor/destructor
	~TrainingParam()
	{
		if (m_pFeatureSpace != NULL) {
			delete m_pFeatureSpace;
		}
	}
	;
	TrainingParam()
	{
		memset(&m_orfp, 0, sizeof(m_orfp));
		m_orfp.numClasses = 2;
		m_orfp.numFeatures = 1024;
		m_orfp.numTrees = 20;
		m_orfp.maxDepth = 7;
		m_orfp.numRandomTests = 16;
		m_orfp.numProjFeatures = 1;
		m_orfp.numEpochs = 1;
		m_orfp.useSoftVoting = 1;

		m_pFeatureSpace = NULL;
		m_iNumPosSamples = 0;
		m_iNumNegSamples = 0;
		m_iHaarStep = 2;
		m_iEpochs = 1000;
		m_patchSize = Size(64, 128);

		m_strOutputFile = "orf.bin";
	}

	// member functions
	void ReadConfigFile(const char *fname);
	void Validate(void);
	void Print(void);

	CFeatureSpace* m_pFeatureSpace;
	CORFParameters m_orfp; // ORF parameters
	String m_featureFile; // Haar space file
	String m_posFile; // positive sample file
	String m_negFile; // negative sample file
	String m_strOutputFile; // output file
	int m_iNumPosSamples; // number of positive samples
	int m_iNumNegSamples; // number of negative samples
	int m_iHaarStep;
	int m_iEpochs;
	Size m_patchSize;
	SampleSet m_trainingSet;
	SampleSet m_validationSet;
} TrainingParam;

/*******************************************************************************
 Generate sample.
 *******************************************************************************/
void GenerateSample(CFeatureSpace* pFeatureSpace, Mat &image, int label, double weight,
		Sample &sample)
{
	sample.w = weight;
	sample.y = label;
	if (image.channels() == 1)
		cvtColor(image, sample.imagePatch, COLOR_GRAY2BGR);
	else
		sample.imagePatch = image;
	pFeatureSpace->ComputeSampleFeatures(sample);
}

void CreateSamples(TrainingParam& par)
{
	char fileName[100];

	// Create positive samples
	FILE *pf = fopen(par.m_posFile.c_str(), "r");
	if (pf != NULL) {
		int count = 0;
		while (!feof(pf)) {
			fscanf(pf, "%s\n", fileName);
			printf("image %s\n", fileName);
			Mat image = imread(fileName, IMREAD_GRAYSCALE);
			Mat resized;
			if (image.cols != par.m_patchSize.width || image.rows != par.m_patchSize.height)
				resize(image, resized, par.m_patchSize);
			else
				resized = image;

			Sample *sample = new Sample(par.m_pFeatureSpace->GetFeatureDimension());
			GenerateSample(par.m_pFeatureSpace, resized, 1, 1.0, *sample);
			par.m_trainingSet.push_back(sample);

			//if (par.m_trainingSet.size() > 200)
			//	break;
			count++;
		}
		fclose(pf);
	} else {
		printf("Cannot open file %s\n", par.m_posFile.c_str());
	}

	// Create negative samples
	pf = fopen(par.m_negFile.c_str(), "r");
	if (pf != NULL) {
		int count = 0;
		while (!feof(pf)) {
			fscanf(pf, "%s\n", fileName);
			printf("image %s\n", fileName);
			Mat image = imread(fileName, IMREAD_GRAYSCALE);
			Mat resized;
			if (image.cols != par.m_patchSize.width || image.rows != par.m_patchSize.height)
				resize(image, resized, par.m_patchSize);
			else
				resized = image;

			Sample *sample = new Sample(par.m_pFeatureSpace->GetFeatureDimension());
			GenerateSample(par.m_pFeatureSpace, resized, 0, 1.0, *sample);
			par.m_trainingSet.push_back(sample);

			//if (par.m_trainingSet.size() > 400)
			//	break;
			count++;
		}
		fclose(pf);
	} else {
		printf("Cannot open file %s\n", par.m_negFile.c_str());
	}

	printf("Training set samples = %ld\n", par.m_trainingSet.size());
	printf("Validation set samples = %ld\n", par.m_validationSet.size());
}

void CreateValidationSet(TrainingParam& par)
{
	vector<int> randIndex;
	RandPerm(par.m_trainingSet.size(), randIndex);
	par.m_validationSet.clear();
	for (int i = 0; i < par.m_trainingSet.size() / 3; i++) {
		par.m_validationSet.push_back(par.m_trainingSet[randIndex[i]]);
	}
}

/*******************************************************************************
 Run validation.
 *******************************************************************************/
int RunValidation(TrainingParam& par, OnlineRF* pRF)
{
	printf("\nRunning validation ...\n");
	int total = par.m_validationSet.size();
	int passed = 0;
	for (int i = 0; i < total; i++) {
		Sample *sample = par.m_validationSet[i];
		Result result;
		pRF->Evaluate(*sample, result);
		if (result.prediction == sample->y) {
			passed++;
		}
	}

	float rate = (float) passed / total;
	int ret = 0;
	if (rate > 0.99f) {
		ret = 1;
	}

	printf("total = %d, passed = %d, rate = %6.3f\n", total, passed, rate);
	return ret;
}

/*
 * Main function.
 */
#if 1
int main(int argc, char *argv[])
{
	const String about = "\n"
			"********************\n"
			"ORF training program\n"
			"********************\n";

	const String keys = "{help h usage ? |                | Print this message   }"
			"{pos            | pos_list.txt   | Name of positive sample list file }"
			"{neg            | neg_list.txt   | Name of negative sample list file }"
			"{feature        | haar.bin       | Name of the feature file }"
			"{epochs         | 10             | Number of epochs to run }";

	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	parser.printMessage();
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	TrainingParam par;
	par.m_posFile = parser.get<String>("pos");
	par.m_negFile = parser.get<String>("neg");

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	// create Haar feature space
	par.m_pFeatureSpace = (CHOGSpace*) new CHOGSpace;
	assert(par.m_pFeatureSpace != NULL);
	par.m_orfp.numFeatures = par.m_pFeatureSpace->GetFeatureDimension();

	// create RF
	OnlineRF* pRF;
	par.m_orfp.numTrees++;
	pRF = new OnlineRF(par.m_orfp);
	assert(pRF != NULL);

	// Create samples
	CreateSamples(par);

	// Training loop
	int run = 0;
	for (int run = 0; run < par.m_iEpochs; run++) {
		printf("\nEpoch %d\n", run);

		// Create training/validation lists
		CreateValidationSet(par);

		// run training
		printf("\nRunning training ...\n");
		pRF->Train(par.m_trainingSet);
		printf("Done\n");

		int result = RunValidation(par, pRF);
		pRF->Analyse();

		if (run < par.m_iEpochs - 1) {
			pRF->Prune(0.01);
		}

		if (result) {
			break;
		}
	}

	// delete last tree and re-run validation
	pRF->DeleteTree(par.m_orfp.numTrees - 1);
	int result = RunValidation(par, pRF);
	pRF->Analyse();
	double oobe;
	pRF->GetMaxTreeOOBE(oobe);

	// Store trained ORF into file
	if (par.m_strOutputFile.length() > 0) {
		FILE *pf = fopen(par.m_strOutputFile.c_str(), "wb");
		if (pf != NULL) {
			par.m_orfp.Store(pf);
			pRF->Store(pf);
			fclose(pf);
		}

		// read ORF from file to verify
		pf = fopen(par.m_strOutputFile.c_str(), "rb");
		if (pf != NULL) {
			delete pRF;
			par.m_orfp.Load(pf);
			par.m_orfp.Print();
			pRF = new OnlineRF(par.m_orfp);
			pRF->Load(pf);
			fclose(pf);
			pRF->Analyse();
		}
	}

	delete pRF;
	return 0;
}

#else

int main0(int argc, char *argv[])
{
	const String about =
	"\n"
	"********************\n"
	"ORF training program\n"
	"********************\n";

	const String keys =
	"{help h usage ? |                | Print this message   }"
	"{pos            | pos_list.txt   | Name of positive sample list file }"
	"{neg            | neg_list.txt   | Name of negative sample list file }"
	"{feature        | haar.bin       | Name of the feature file }"
	"{epochs         | 10             | Number of epochs to run }"
	;

	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	parser.printMessage();
	if (parser.has("help")) {
		parser.printMessage();
		return 0;
	}

	TrainingParam par;
	par.m_featureFile = parser.get<String>("feature");
	par.m_posFile = parser.get<String>("pos");
	par.m_negFile = parser.get<String>("neg");

	if (!parser.check()) {
		parser.printErrors();
		return 0;
	}

	// create Haar feature space
	par.m_pHaarSpace = new CHaarSpace(NULL);
	assert(par.m_pHaarSpace != NULL);

	// load or initialize Haar space
	FILE* pfHaar = fopen(par.m_featureFile.c_str(), "rb");
	if (pfHaar != NULL) {
		if (!par.m_pHaarSpace->Load(pfHaar)) {
			par.m_pHaarSpace->InitFeatures(par.m_iHaarStep);
		}
	} else {
		par.m_pHaarSpace->InitFeatures(par.m_iHaarStep);
		pfHaar = fopen(par.m_featureFile.c_str(), "wb");
		if (pfHaar != NULL) {
			par.m_pHaarSpace->Store(pfHaar);
			fclose(pfHaar);
			pfHaar = NULL;
		}
	}
	par.m_orfp.numFeatures = par.m_pHaarSpace->GetNumberFeatures();
	printf("Number of Haar features = %d\n", par.m_orfp.numFeatures);

	// create RF
	OnlineRF* pRF;
	pRF = new OnlineRF(par.m_orfp);
	assert(pRF != NULL);

	// Create samples
	CreateSamples(par);

	// Training loop
	int run = 0;
	for (int run = 0; run < par.m_iEpochs; run++) {
		printf("\nEpoch %d\n", run);

		// Create training/validation lists

		// run training
		printf("\nRunning training ...\n");
		pRF->Train(par.m_trainingSet);
		printf("Done\n");

		int result = RunValidation(par, pRF);

		pRF->Analyse();
		//pRF->GetMaxTreeOOBE();
		if (result) {
			break;
		}
	}

	// Store trained ORF into file
	if (par.m_strOutputFile.length() > 0) {
		FILE *pf = fopen(par.m_strOutputFile.c_str(), "wb");
		if (pf != NULL) {
			par.m_orfp.Store(pf);
			pRF->Store(pf);
			fclose(pf);
		}
	}

	delete pRF;

	return 0;
}

#endif
