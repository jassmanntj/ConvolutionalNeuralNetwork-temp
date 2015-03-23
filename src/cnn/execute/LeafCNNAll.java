package cnn.execute;

/**
 * Created by jassmanntj on 3/20/2015.
 */
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;

import cnn.*;
import org.jblas.DoubleMatrix;

public class LeafCNNAll {
    private static final boolean load = false;
    DoubleMatrix images = null;
    DoubleMatrix labels = null;
    DoubleMatrix testImages = null;
    DoubleMatrix testLabels = null;
    double[] correctTest;
    double[] correctTrain;
    int totalTest;
    int totalTrain;

    public static void main(String[] args) throws Exception {
        new LeafCNNAll().run();
    }

    public LeafCNNAll() {
        correctTest = new double[3];
        correctTrain = new double[3];
        totalTest = 0;
        totalTrain = 0;
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
        int iterations = 500;
        boolean loaded = false;

        ImageLoader loader = new ImageLoader();
        File folder = new File("C:/Users/jassmanntj/Desktop/CA-Leaves");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        for(int i = 0; i < 30; i++) {
            String resFile = numPatches+"-Cross-"+patchSize+"-"+poolSize+"-"+iterations+"."+i;
            if (!load || !(new File("data/TrainImgs-"+resFile+".mat").exists())) {
                if(!loaded) {
                    loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
                    loaded = true;
                }
                loader.sortImgs(i);
                images = loader.getImgs();
                labels = loader.getLbls();
                testImages = loader.getTestImgs();
                testLabels = loader.getTestLbls();
                images.save("data/TrainImgs-"+resFile+".mat");
                labels.save("data/TrainLbls-"+resFile+".mat");
                testImages.save("data/TestImgs-"+resFile+".mat");
                testLabels.save("data/TestLbls-"+resFile+".mat");
            }
            else {
                images = new DoubleMatrix("data/TrainImgs-"+resFile+".mat");
                labels = new DoubleMatrix("data/TrainLbls-"+resFile+".mat");
                testImages = new DoubleMatrix("data/TestImgs-"+resFile+".mat");
                testLabels = new DoubleMatrix("data/TestLbls-"+resFile+".mat");
            }

            ConvolutionLayer cl = new ConvolutionLayer(channels, patchSize, imageRows, imageColumns, poolSize, numPatches, sparsityParam, lambda, beta, alpha, true);
            SparseAutoencoder ae2 = new SparseAutoencoder(cl.getOutputSize(), hiddenSize, cl.getOutputSize(), sparsityParam, lambda, beta, alpha);
            SoftmaxClassifier sc = new SoftmaxClassifier(1e-4);
            SparseAutoencoder[] saes = {ae2};
            DeepNN dn = new DeepNN(saes, sc);
            NeuralNetworkLayer[] nnl = {cl, dn};
            ConvolutionalNeuralNetwork cnn = new ConvolutionalNeuralNetwork(nnl, resFile);

            DoubleMatrix result = cnn.train(images, labels, iterations);
            int[][] results = Utils.computeResults(result);

            DoubleMatrix testRes = cnn.computeRes(testImages);
            int[][] testResult = Utils.computeResults(testRes);
            System.out.println("TRAIN");
            compareResults(results, labels, false);
            System.out.println("TEST");
            compareResults(testResult, testLabels, true);
            compareClasses(testResult, testLabels, labelMap);
        }

    }

    public void compareResults(int[][] result, DoubleMatrix labels, boolean test) {
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
        if(test) {
            correctTest[0] += sum1;
            correctTest[1] += sum2;
            correctTest[2] += sum3;
            totalTest += result.length;
            System.out.println("OVERALL");
            System.out.println("First: "+correctTest[0]+"/"+totalTest+ " = " + (correctTest[0]/totalTest));
            System.out.println("Second: "+correctTest[1]+"/"+totalTest+ " = " + (correctTest[1]/totalTest));
            System.out.println("Third: "+correctTest[2]+"/"+totalTest+ " = " + (correctTest[2]/totalTest));
        }
        else {
            correctTrain[0] += sum1;
            correctTrain[1] += sum2;
            correctTrain[2] += sum3;
            totalTrain += result.length;
            System.out.println("OVERALL");
            System.out.println("First: "+correctTrain[0]+"/"+totalTrain+ " = " + (correctTrain[0]/totalTrain));
            System.out.println("Second: "+correctTrain[1]+"/"+totalTrain+ " = " + (correctTrain[1]/totalTrain));
            System.out.println("Third: "+correctTrain[2]+"/"+totalTrain+ " = " + (correctTrain[2]/totalTrain));
        }
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
            System.out.println(newMap.get((double)i)+": " +count[i]+"/"+totalCount[i]+" = "+(count[i]/totalCount[i]));
        }
    }
}
