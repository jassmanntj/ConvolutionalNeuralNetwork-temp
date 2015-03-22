package cnn.execute;
/** Try 2 layer 100 patch for d
 * Alla - 100 patch all 800x600 ?
 * Allb - 100 patch all 90% 70%
 * Allc - 200 patch all 99% 79%
 * Alld - 100 patch cl 200 sae sc 99.88% 79.28%
 * Alle - 50 patch cl 200 sae sc 99.77% 79.10%
 * Allf - ^^ 6 patchsize   100% 79.8% 4.4s
 * AllAll - 200 8 patch cl 200 sae sc 90% 72%
 * AllAll2 - 200 8 patch cl 200 sae 200 sae sc  -- bad
 * AllAll3 - 200 8 patch cl 200 sae sc 99.9% 79.9% 800 iterations 36.244s
 * AllAll4 - same 6 patch 800 iterations 95.4% 76.2% 
 * AllAll5 - same 100 6 patch 3200 iterations 99.9% 80.1% 9.204s
 * AllAll6 - 50 6 patch 99.9% 79.7% 4.866s
 * AllAll7 - 25 6 patch 99.4% 79.2% 3.270s
 * AllAll5.6 - 25 6 patch 5 pool 100% 80.4% 2.578s
 * AllAll5.6.1 - 16 6 patch 5 pool 100% 80.3% 2.152s
 * 
 * ========================================================================
 * 
 * Final11.8.10.200 - xxxx xxxx
 * Final11.64.10.200 - 100% 28/7/5% 5/5/4% err saving
 * Final11.32.10.200
 *
 * =========================================================================
 128 5x4
 TRAIN
 First: 404.0/404 = 1.0
 Second: 0.0/404 = 0.0
 Third: 0.0/404 = 0.0
 TEST
 First: 29.0/45 = 0.6444444444444445
 Second: 8.0/45 = 0.17777777777777778
 Third: 3.0/45 = 0.06666666666666667

 32 5x4
 TRAIN
 First: 404.0/404 = 1.0
 Second: 0.0/404 = 0.0
 Third: 0.0/404 = 0.0
 TEST
 First: 27.0/45 = 0.6
 Second: 3.0/45 = 0.06666666666666667
 Third: 5.0/45 = 0.1111111111111111

 32 6x5
 TRAIN
 First: 404.0/404 = 1.0
 Second: 0.0/404 = 0.0
 Third: 0.0/404 = 0.0
 TEST
 First: 24.0/45 = 0.5333333333333333
 Second: 8.0/45 = 0.17777777777777778
 Third: 3.0/45 = 0.06666666666666667
 *
 *
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

import org.jblas.DoubleMatrix;

import cnn.ConvolutionLayer;
import cnn.ConvolutionalNeuralNetwork;
import cnn.DeepNN;
import cnn.ImageLoader;
import cnn.NeuralNetworkLayer;
import cnn.SoftmaxClassifier;
import cnn.SparseAutoencoder;
import cnn.Utils;

public class LeafCNN {
	private static final boolean load = false;
	DoubleMatrix images;
	DoubleMatrix photoTrain;
	DoubleMatrix labels;
	DoubleMatrix photoLabels;
	DoubleMatrix testImages;
	DoubleMatrix testLabels;
	DoubleMatrix photoTest;
	DoubleMatrix photoTestLabels;

	public static void main(String[] args) throws Exception {
		new LeafCNN().run();
	}

	public void run() throws Exception {
		int patchSize = 5;
		int poolSize = 4;
		int numPatches = 64;
		//int hiddenSize = 200;
		int imageRows = 80;
		int imageColumns = 60;
		double sparsityParam = 0.035;
		double lambda = 3e-3;
		double beta = 5;
		double alpha = 0.5;
		int channels = 3;
        int hiddenSize = 250;

        //CL2
        int patchSize2 = 2;
        int poolSize2 = 2;
        int numPatches2 = 64;
        int imageRows2 = (imageRows-patchSize+1)/poolSize;
        int imageColumns2 = (imageColumns-patchSize+1)/poolSize;
        int channels2 = numPatches;

        String resFile = "Final"+patchSize+"."+numPatches+"."+poolSize+"."+patchSize2+"."+numPatches2+"."+poolSize2+".1";
		
		ImageLoader loader = new ImageLoader();
		File folder = new File("C:/Users/jassmanntj/Desktop/CA-Leaves-train");
		HashMap<String, Double> labelMap = loader.getLabelMap(folder);
		
		if(!load) {
			loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
			images = loader.getImages();
			//photoTrain = loader.getPhotoImages();
			labels = loader.getLabels();
			//photoLabels = loader.getPhotoLabels();
			//labels.save("data/TrainLabels.mat");
			//photoLabels.save("data/PhotoTrainLabels.mat");
			//photoTrain.save("data/PhotoTrain.mat");
			//images.save("data/Train.mat");
		}
		else {
			labels = new DoubleMatrix("data/TrainLabels.mat");
			photoLabels = new DoubleMatrix("data/PhotoTrainLabels.mat");
			photoTrain = new DoubleMatrix("data/PhotoTrain.mat");
			images = new DoubleMatrix("data/Train.mat");
		}

		ConvolutionLayer cl = new ConvolutionLayer(channels, patchSize, imageRows, imageColumns, poolSize, numPatches, sparsityParam, lambda, beta, alpha, true);
		//ConvolutionLayer cl2 = new ConvolutionLayer(channels2, patchSize2, imageRows2, imageColumns2, poolSize2, numPatches2, sparsityParam, lambda, beta, alpha, true);
		SparseAutoencoder ae2 = new SparseAutoencoder(cl.getOutputSize(), hiddenSize, cl.getOutputSize(), sparsityParam, lambda, beta, alpha);
		SoftmaxClassifier sc = new SoftmaxClassifier(1e-4);
		SparseAutoencoder[] saes = {ae2};
		DeepNN dn = new DeepNN(saes, sc);
		NeuralNetworkLayer[] nnl = {cl, dn};
		ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(nnl, resFile);
		
		DoubleMatrix result = cnn.train(images, labels, 200);
		int[][] results = Utils.computeResults(result);
		System.out.println("TRAIN");
		

		folder = new File("C:/Users/jassmanntj/Desktop/CA-Leaves-test");
		if(!load) {
			loader.loadFolder(folder,channels,imageColumns, imageRows,labelMap);
			testImages = loader.getImages();
			testLabels = loader.getLabels();
			//photoTest = loader.getPhotoImages();
			//photoTestLabels = loader.getPhotoLabels();
			//photoTest.save("data/PhotoTest.mat");
			//photoTestLabels.save("data/PhotoTestLabels.mat");
			//testImages.save("data/Test.mat");
			//testLabels.save("data/TestLabels.mat");
		}
		else {
			photoTest = new DoubleMatrix("data/PhotoTest.mat");
			photoTestLabels = new DoubleMatrix("data/PhotoTestLabels.mat");
			testImages = new DoubleMatrix("data/Test.mat");
			testLabels = new DoubleMatrix("data/TestLabels.mat");
		}
		
		DoubleMatrix testRes = cnn.computeRes(testImages);
		int[][] testResult = cnn.compute(testImages);
		//DoubleMatrix photoTestRes = cnn.computeRes(photoTest);
		//int[][] photoTestResult = cnn.compute(photoTest);
		System.out.println("TRAIN");
		compareResults(results, labels);
		System.out.println("TEST");
		compareResults(testResult, testLabels);
		/*ArrayList<String> fileNames = loader.getNames(folder);
		ArrayList<String> photoFiles = loader.getPhotoNames();
		writeResults(testRes, testLabels, fileNames, resFile);
		System.out.println("PHOTOTEST");
		compareResults(photoTestResult, photoTestLabels);
		writeResults(photoTestRes, photoTestLabels, photoFiles, "Photo"+resFile);*/
	}
	
	public void compareResults(int[][] result, DoubleMatrix labels) {
		double sum1 = 0;
		double sum2 = 0;
		double sum3 = 0;
		for(int i = 0; i < result.length; i++) {
			sum1 += labels.get(i, result[i][0]);
			sum2 += labels.get(i, result[i][1]);
			sum3 += labels.get(i, result[i][2]);
		}
		System.out.println("First: "+sum1+"/"+result.length+ " = " + (sum1/result.length));
		System.out.println("Second: "+sum2+"/"+result.length+ " = " + (sum2/result.length));
		System.out.println("Third: "+sum3+"/"+result.length+ " = " + (sum3/result.length));		
	}
	
	private HashMap<Double, String> reverseMap(HashMap<String, Double> map) {
		HashMap<Double, String> newMap = new HashMap<Double, String>();
		for(String key : map.keySet()) {
			newMap.put(map.get(key), key);
		}
		return newMap;
	}
	
	public void writeResults(DoubleMatrix results, DoubleMatrix labels, ArrayList<String> fileNames, String filename) {
		try {
			FileWriter fw = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fw);
			for(int i = 0; i < results.rows; i++) {
				DoubleMatrix row = results.getRow(i);
				for(int j = 0; j < row.columns; j++) {
					if(labels.get(i, row.argmax())==1) {
						writer.write(fileNames.get(i).replace(".*", ".xml")+";"+j+"\n");
						break;
					}
					row.put(0, row.argmax(),-1);
				}
			}
			writer.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	public void compareClasses(int[][] result, DoubleMatrix labels, HashMap<String, Double> labelMap) throws Exception {
		double[] totalCount = new double[labels.columns];
		double[] count = new double[labels.columns];
		HashMap<Double, String> newMap = reverseMap(labelMap);
		for(int i = 0; i < result.length; i++) {
			int labelNo = -1;
			for(int j = 0; j < labels.columns; j++) {
				if((int)labels.get(i, j) == 1) {
					if(labelNo == -1) {
						labelNo = j;
					}
					else throw new Exception("Invalid Labels");
				}
			}
			if(labelNo == result[i][0]) {
				count[labelNo]++;
			}
			totalCount[labelNo]++;
		}
		for(int i = 0; i < count.length; i++) {
			System.out.println(newMap.get((double)i)+": " + (count[i]/totalCount[i]));
		}	
	}
}
