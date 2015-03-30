package cnn;

import org.jblas.DoubleMatrix;

import java.io.File;
import java.io.IOException;
import java.util.Random;

/**
 * Created by Tim on 3/29/2015.
 */
public class CNN2 {
    private NeuralNetworkConvLayer[] cls;
    private LinearDecoder[] lds;
    private SoftmaxClassifier sc;
    private String name;
    private Random r;

    public CNN2(NeuralNetworkConvLayer[] cls, LinearDecoder[] lds, SoftmaxClassifier sc, String name) {
        this.cls = cls;
        this.lds = lds;
        this.sc = sc;
        this.name = name;
        r = new Random(System.currentTimeMillis());
    }

    public DoubleMatrix train(DoubleMatrix[][] input, DoubleMatrix labels, int iterations) throws IOException {
        int batchSize = 30;
        int[] indices = new int[input.length];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        for(int i = 0; i < iterations; i++) {
            shuffleIndices(indices, indices.length);
            for(int j = 0; j < input.length/batchSize; j++) {
                DoubleMatrix[][] in = new DoubleMatrix[input.length/batchSize][];
                for(int k = 0; k < in.length; k++) {
                    in[k] = input[indices[j*batchSize+k]];
                }
                for(int k = 0; k < cls.length; k++) {
                    in = cls[k].compute(in);
                }
                DoubleMatrix in2 = Utils.flatten(in);
                for(int k = 0; k < lds.length; k++) {
                    in2 = lds[k].compute(in2);
                }
                CostResult cr = sc.stackedCost(in2, labels);
                System.out.println("Cost: "+cr.cost);
                DoubleMatrix delta = sc.backpropagation(cr);
                for(int k = 0; k < lds.length; k++) {

                }
                for(int k = 0; k < cls.length; k++) {
                    cls[k].backPropagation()
                }


            }
        }


        for(int i = 0; i < layers.length; i++) {
            File f = new File(name+"Layer"+i+".layer");
            if(f.exists() || new File(name+"Layer"+i+".layer0").exists()) {
                System.out.println("ALayer"+i);
                layers[i].loadLayer(name+"Layer"+i+".layer");
                input = layers[i].compute(input);
            }
            else {
                System.out.println("BLayer"+i);
                input = layers[i].train(input, labels, iterations);
                layers[i].writeLayer(name+"Layer"+i+".layer");
            }
        }
        return input;
    }

    private void shuffleIndices(int[] indices, int iterations) {
        for(int i = 0; i < iterations; i++) {
            int a = r.nextInt(indices.length);
            int b = r.nextInt(indices.length);
            int temp = indices[a];
            indices[a] = indices[b];
            indices[b] = temp;
        }
    }

    public int[][] compute(DoubleMatrix input) {
        for(int i = startLayer; i < layers.length; i++) {
            input = layers[i].compute(input);
        }
        return Utils.computeResults(input);
    }

    public DoubleMatrix computeRes(DoubleMatrix input) {
        for(int i = startLayer; i < layers.length; i++) {
            input = layers[i].compute(input);
        }
        return input;
    }

}
