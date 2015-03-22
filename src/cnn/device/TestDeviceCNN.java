package cnn.device;

import java.io.File;
import java.io.IOException;

import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class TestDeviceCNN {
	public static void main(String[] args) throws IOException {
		new TestDeviceCNN().run();
	}
	
	public void run() throws IOException {
		DeviceImageLoader loader = new DeviceImageLoader();
		
		DeviceConvolutionLayer cl = new DeviceConvolutionLayer("LeafCNNaLayer1.layer");
		DeviceSoftmaxClassifier sc = new DeviceSoftmaxClassifier("LeafCNNaLayer2.layer");
		DeviceNeuralNetworkLayer[] nnl = {cl, sc};
		DeviceCNN cnn = new DeviceCNN(nnl);
		
		loader.loadFolder(new File("C:/Users/jassmanntj/Desktop/Images"), 3, 60, 80);
		DenseDoubleMatrix2D images = loader.getImages();
		DenseDoubleMatrix2D labels = loader.getLabels();
		int[] results = cnn.compute(images);
		
		//loader.loadFolder(new File("C:/Users/jassmanntj/Desktop/TestImages"),3,60,80);
		//DenseDoubleMatrix2D testImages = loader.getImages();
		//DenseDoubleMatrix2D testLabels = loader.getLabels();
		//int[] testResult = cnn.compute(testImages);
		
		System.out.println("TRAIN");
		compareResults(results, labels);
		
		//System.out.println("TEST");
		//compareResults(testResult, testLabels);
		
	}
	
	public void compareResults(int[] result, DenseDoubleMatrix2D labels) {
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += labels.get(i, result[i]);
		}
		System.out.println(sum+"/"+result.length);
		System.out.println(sum/result.length);
	}
}

