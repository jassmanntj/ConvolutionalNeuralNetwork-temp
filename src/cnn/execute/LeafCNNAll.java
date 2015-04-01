package cnn.execute;

/**
 * Created by jassmanntj on 3/20/2015.
 */
/*import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import cnn.*;
import org.jblas.DoubleMatrix;

public class LeafCNNAll {
    private static final boolean load = true;
    DoubleMatrix images = null;
    DoubleMatrix labels = null;
    DoubleMatrix testImages = null;
    DoubleMatrix testLabels = null;
    double[] correctTest;
    double[] correctTrain;
    double[] count;
    double[] totalCount;
    int totalTest;
    int totalTrain;
    int numPatches = 20;

    public static void main(String[] args) throws Exception {
        new LeafCNNAll().run();
    }

    public void run() throws Exception {
        int patchSize = 5;
        int poolSize = 4;
        //int hiddenSize = 200;
        int imageRows = 80;
        int imageColumns = 60;
        double sparsityParam = 0.035;
        double lambda = 3e-3;
        double beta = 5;
        double alpha = 0.5;
        int channels = 3;
        int hiddenSize = 250;
        int iterations = 200;
        boolean loaded = false;

        ImageLoader loader = new ImageLoader();
        File folder = new File("C:/Users/jassmanntj/Desktop/CA-Leaves");
        HashMap<String, Double> labelMap = loader.getLabelMap(folder);
        while(numPatches < 300) {
            count = new double[labelMap.size()];
            totalCount = new double[labelMap.size()];
            correctTest = new double[labelMap.size()];
            correctTrain = new double[labelMap.size()];
            totalTest = 0;
            totalTrain = 0;
            for (int i = 0; i < 30; i += 3) {
                String resFile = numPatches + "-Cross-" + patchSize + "-" + poolSize + "-" + iterations + "." + i;
                if (!load || !(new File("data/TrainImgs-" + resFile + ".mat").exists())) {
                    if (!loaded) {
                        loader.loadFolder(folder, channels, imageColumns, imageRows, labelMap);
                        loaded = true;
                    }
                    loader.sortImgs(i);
                    images = loader.getImgs();
                    labels = loader.getLbls();
                    testImages = loader.getTestImgs();
                    testLabels = loader.getTestLbls();
                    images.save("data/TrainImgs-" + resFile + ".mat");
                    labels.save("data/TrainLbls-" + resFile + ".mat");
                    testImages.save("data/TestImgs-" + resFile + ".mat");
                    testLabels.save("data/TestLbls-" + resFile + ".mat");
                } else {
                    images = new DoubleMatrix("data/TrainImgs-" + resFile + ".mat");
                    labels = new DoubleMatrix("data/TrainLbls-" + resFile + ".mat");
                    testImages = new DoubleMatrix("data/TestImgs-" + resFile + ".mat");
                    testLabels = new DoubleMatrix("data/TestLbls-" + resFile + ".mat");
                }

                ConvolutionLayer cl = new ConvolutionLayer(channels, patchSize, imageRows, imageColumns, poolSize, numPatches, sparsityParam, lambda, beta, alpha, true);
                SparseAutoencoder ae2 = new SparseAutoencoder(cl.getOutputSize(), hiddenSize, cl.getOutputSize(), sparsityParam, lambda, beta, alpha);
                SoftmaxClassifier sc = new SoftmaxClassifier(1e-4);
                SparseAutoencoder[] saes = {ae2};
                DeepNN dn = new DeepNN(saes, sc);
                NeuralNetworkLayer[] nnl = {cl, dn};
                NeuralNetwork cnn = new NeuralNetwork(nnl, "data/" + resFile);
                System.out.println("LABELS:" + labels.rows + "x" + labels.columns);
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
            numPatches += 10;
        }

    }

    public void compareResults(int[][] result, DoubleMatrix labels, boolean test) throws IOException {
        FileWriter fw = new FileWriter("data/ConvolutionSizeResults", true);
        BufferedWriter bw = new BufferedWriter(fw);
        double[] sums = new double[result[0].length];

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                sums[j] += labels.get(i, result[i][j]);
            }
        }
        for(int i = 0; i < sums.length; i++) {
            if(sums[i] > 0) System.out.println(i+": "+sums[i]+"/"+result.length+ " = " + (sums[i]/result.length));
        }
        if(test) {
            System.out.println("OVERALL");
            totalTest += result.length;
            if(totalTest==420) bw.write(numPatches+"\n");
            for(int i = 0; i < sums.length; i++) {
                correctTest[i] += sums[i];
                if(correctTest[i] > 0) {
                    System.out.println(i+": "+correctTest[i]+"/"+totalTest+ " = " + (correctTest[i]/totalTest));
                    if(totalTest==420) bw.write(i+": "+correctTest[i]+"/"+totalTest+ " = " + (correctTest[i]/totalTest)+"\n");
                }
            }
        }
        else {
            System.out.println("OVERALL");
            totalTrain += result.length;
            for(int i = 0; i < sums.length; i++) {
                correctTrain[i] += sums[i];
                if(correctTrain[i] > 0) System.out.println(i+": "+correctTrain[i]+"/"+totalTrain+ " = " + (correctTrain[i]/totalTrain));
            }
        }
        bw.close();
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
}*/
