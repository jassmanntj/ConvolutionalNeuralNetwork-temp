/*package cnn.device;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import org.jblas.DoubleMatrix;

import cnn.ConvolutionLayer;
import cnn.ImageLoader;
import cnn.LinearDecoder;
import cnn.SoftmaxClassifier;
import cnn.Utils;
import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class TestDeviceSoftmax {
	public static void main(String[] args) throws IOException {
		DeviceImageLoader loader = new DeviceImageLoader();
		loader.loadFolder(new File("C:/Users/jassmanntj/Desktop/Images"), 3, 60, 80);
		DenseDoubleMatrix2D labels = loader.getLabels();
		ImageLoader l = new ImageLoader();
		File folder = new File("C:/Users/jassmanntj/Desktop/Images");
		HashMap<String, Double> labelMap = l.getLabelMap(folder);
		l.loadFolder(folder, 3, 60, 80, labelMap);
		DoubleMatrix patches = new DoubleMatrix(100000, 8*8*3);
		patches.load("Patches8.csv");
		patches.divi(patches.max());
		DoubleMatrix meanPatch = patches.columnMeans();
		DoubleMatrix ZCAWhite = Utils.calculateZCAWhite(patches, meanPatch, 0.1);
		DoubleMatrix images = l.getImages();
		patches = Utils.ZCAWhiten(patches, meanPatch, ZCAWhite);
		LinearDecoder ae = new LinearDecoder(8, 3, 400, .035, 3e-3, 5, .5);
		ConvolutionLayer c = new ConvolutionLayer(ZCAWhite, meanPatch, images, ae, 3, 8, 60, 80, 19);
		DoubleMatrix features = c.loadTheta("LeafCNN5OutLayer1.csv", null);
		DeviceSoftmaxClassifier sc = new DeviceSoftmaxClassifier("LeafCNN5OutLayer2.csv");
		DenseDoubleMatrix2D f = new DenseDoubleMatrix2D(features.toArray2());
		System.out.println(f.get(1, 1)+":"+features.get(1,1));
		sc.loadTheta("LeafCNN5OutLayer2.csv");
		SoftmaxClassifier s = new SoftmaxClassifier(1e-4);
		s.loadTheta("LeafCNN5OutLayer2.csv");
		DoubleMatrix theta1 = s.getTheta();
		DoubleMatrix2D theta2 = sc.getTheta();
		System.out.println(theta1.get(0,0)+":"+theta2.get(0, 0)+"\n"+theta1.get(0,1)+":"+theta2.get(0, 1)+"\n"+theta1.get(1,0)+":"+theta2.get(1, 0));
		DenseDoubleMatrix2D result = sc.compute(f);
		DoubleMatrix r = s.compute(features);
		int[] res = DeviceUtils.computeResults(result);
		int[][] r2 = Utils.computeResults(r);
		compareResults(res, labels);
		compareResults(r2, labels);
	}
	
	public static void compareResults(int[][] result, DenseDoubleMatrix2D labels) {
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += labels.get(i, result[i][0]);
		}
		System.out.println(sum+"/"+result.length);
		System.out.println(sum/result.length);
	}
	public static void compareResults(int[] result, DenseDoubleMatrix2D labels) {
		double sum = 0;
		for(int i = 0; i < result.length; i++) {
			sum += labels.get(i, result[i]);
		}
		System.out.println(sum+"/"+result.length);
		System.out.println(sum/result.length);
	}
}
*/