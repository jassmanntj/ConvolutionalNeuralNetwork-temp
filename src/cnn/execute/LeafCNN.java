package cnn.execute;

import cnn.*;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.util.HashMap;

/**
 * Created by Tim on 4/1/2015.
 */
public class LeafCNN {

    public static void main(String[] args) throws Exception {
        new LeafCNN().run();
    }

    public void run() throws Exception {
        DoubleMatrix[][] images;
        DoubleMatrix labels;
        int patchDim = 5;
        int poolDim = 4;
        int numFeatures = 64;
        //int hiddenSize = 200;
        int imageRows = 80;
        int imageColumns = 60;
        double sparsityParam = 0.035;
        double lambda = 3e-3;
        double beta = 5;
        double alpha =  1e-3;
        int channels = 3;
        int hiddenSize = 250;
        int batchSize = 30;
        double momentum = 0.9;

        String resFile = "Testing";

        ImageLoader loader = new ImageLoader();
        File folder = new File("C:\\Users\\Tim\\Desktop\\CA-Leaves");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
        images = loader.getImgArr();
        labels = loader.getLabels();

        ConvolutionLayer cl1 = new ConvolutionLayer(numFeatures, channels, patchDim);
        PoolingLayer cl2 = new PoolingLayer(poolDim);

        SparseAutoencoder sa = new SparseAutoencoder(numFeatures * ((imageRows-patchDim+1)/poolDim) * ((imageColumns - patchDim+1)/poolDim), hiddenSize, sparsityParam, lambda, beta, alpha);
        SparseAutoencoder sa2 = new SparseAutoencoder(hiddenSize, 100, sparsityParam, lambda, beta, alpha);

        SoftmaxClassifier sc = new SoftmaxClassifier(1e-4, hiddenSize, labels.columns);
        SparseAutoencoder[] saes = {sa};
        ConvPoolLayer[] cls = {cl1, cl2};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, sc, resFile);
        cnn.train(images, labels, 200, 30, momentum, alpha);
    }
}
