package cnn.execute;

import cnn.*;
import org.jblas.DoubleMatrix;

/**
 * Created by Tim on 3/31/2015.
 */
public class LeafCNNGradientChecking {

    public static void main(String[] args) throws Exception {
        new LeafCNNGradientChecking().run();
    }

    public void run() throws Exception {
        DoubleMatrix[][] images;
        DoubleMatrix labels;
        int patchDim = 3;
        int poolDim = 2;
        int numFeatures = 1;
        //int hiddenSize = 200;
        int imageRows = 10;
        int imageColumns = 10;
        double sparsityParam = 0.035;
        double lambda = 3e-3;
        double beta = 5;
        double alpha =  1e-6;
        int channels = 3;
        int hiddenSize = 150;
        int batchSize = 30;
        double momentum = 0.9;

        String resFile = "Testing";
        images = new DoubleMatrix[2][channels];
        for(int i = 0; i < channels; i++) {
            images[0][i] = DoubleMatrix.rand(10, 10);
        }
        for(int i = 0; i < channels; i++) {
            images[1][i] = DoubleMatrix.rand(10, 10);
        }
        labels = new DoubleMatrix(2,2);
        labels.put(0,0,1);
        labels.put(1,1,1);
        //ImageLoader loader = new ImageLoader();
        //File folder = new File("C:\\Users\\Tim\\Desktop\\CA-Leaves2");
        //HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        //loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
        //images = loader.getImgArr();
        //labels = loader.getLabels();

        ConvolutionLayer cl1 = new ConvolutionLayer(numFeatures, channels, patchDim);
        PoolingLayer cl2 = new PoolingLayer(poolDim);

        SparseAutoencoder sa = new SparseAutoencoder(numFeatures * ((imageRows-patchDim+1)/poolDim) * ((imageColumns - patchDim+1)/poolDim), hiddenSize, sparsityParam, lambda, beta, alpha);
        SparseAutoencoder sa2 = new SparseAutoencoder(hiddenSize, 100, sparsityParam, lambda, beta, alpha);

        SoftmaxClassifier sc = new SoftmaxClassifier(1e-4, hiddenSize, labels.columns);
        SparseAutoencoder[] saes = {sa};
        ConvPoolLayer[] cls = {cl1, cl2};
        NeuralNetwork cnn = new NeuralNetwork(cls, saes, sc, resFile);
        cnn.train(images, labels, 1, 1, momentum, alpha);
        cnn.gradientCheck(images, labels);
    }
}
