package cnn;

import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.Random;

/**
 * Created by Tim on 3/29/2015.
 */
public class NeuralNetwork {
    private ConvPoolLayer[] cls;
    private SparseAutoencoder[] lds;
    private SoftmaxClassifier sc;
    private String name;
    private Random r;

    public NeuralNetwork(ConvPoolLayer[] cls, SparseAutoencoder[] lds, SoftmaxClassifier sc, String name) {
        this.cls = cls;
        this.lds = lds;
        this.sc = sc;
        this.name = name;
        r = new Random(System.currentTimeMillis());
    }

    public void train(DoubleMatrix[][] input, DoubleMatrix labels, int iterations, int batchSize, double momentum, double alpha) throws IOException {
        compareResults(Utils.computeResults(compute(input)), labels);
        int[] indices = new int[input.length];
        for(int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        for(int i = 0; i < iterations; i++) {
            shuffleIndices(indices, indices.length);
            double cost = 0;
            for(int j = 0; j < input.length/batchSize; j++) {
                System.out.println(j);
                DoubleMatrix[][] in = new DoubleMatrix[batchSize][];
                DoubleMatrix labs = new DoubleMatrix(batchSize, labels.columns);
                for(int k = 0; k < in.length; k++) {
                    in[k] = input[indices[j*batchSize+k]];
                    labs.putRow(k, labels.getRow(indices[j*batchSize+k]));
                }
                DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
                convResults[0] = in;
                for(int k = 0; k < cls.length; k++) {
                    convResults[k+1] = cls[k].compute(convResults[k]);
                }
                DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
                ldsResults[0] = Utils.flatten(convResults[cls.length]);
                for(int k = 0; k < lds.length; k++) {
                    ldsResults[k+1] = lds[k].compute(ldsResults[k]);
                }
                CostResult cr = sc.stackedCost(ldsResults[lds.length], labs);
                cost += cr.cost;
                DoubleMatrix delta = sc.backpropagation(cr, momentum, alpha);
                for(int k = lds.length-1; k >= 0; k--) {
                    cr = lds[k].cost(ldsResults[k], ldsResults[k+1], delta);
                    delta = lds[k].backpropagation(cr, momentum, alpha);
                }
                DoubleMatrix[][] delt = expand(delta, convResults[cls.length][0].length, convResults[cls.length][0][0].rows, convResults[cls.length][0][0].columns);
                for(int k = cls.length-1; k >= 0; k--) {
                    delt = cls[k].backPropagation(convResults[k], convResults[k+1], delt, momentum, alpha);
                }
            }
            System.out.println("Iteration "+i+" Cost: " + cost);
            compareResults(Utils.computeResults(compute(input)), labels);
        }
    }

    public void compareResults(int[][] result, DoubleMatrix labels) {
        double[] sums = new double[result[0].length];

        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[i].length; j++) {
                sums[j] += labels.get(i, result[i][j]);
            }
        }
        for(int i = 0; i < sums.length; i++) {
            if(sums[i] > 0) System.out.println(i+": "+sums[i]+"/"+result.length+ " = " + (sums[i]/result.length));
        }
    }

    public CostResult computeCost(DoubleMatrix[][] input, DoubleMatrix labels) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
        convResults[0] = input;
        for(int k = 0; k < cls.length; k++) {
            convResults[k+1] = cls[k].compute(convResults[k]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
        ldsResults[0] = Utils.flatten(convResults[cls.length]);
        for(int k = 0; k < lds.length; k++) {
            ldsResults[k+1] = lds[k].compute(ldsResults[k]);
        }
        CostResult cr = sc.stackedCost(ldsResults[lds.length], labels);
        return cr;
    }

    public void gradientCheck(DoubleMatrix[][] input, DoubleMatrix labels) {
        DoubleMatrix[][][] convResults = new DoubleMatrix[cls.length+1][][];
        convResults[0] = input;
        for(int i = 0; i < cls.length; i++) {
            convResults[i+1] = cls[i].compute(convResults[i]);
        }
        DoubleMatrix[] ldsResults = new DoubleMatrix[lds.length+1];
        ldsResults[0] = Utils.flatten(convResults[cls.length]);
        for(int i = 0; i < lds.length; i++) {
            ldsResults[i+1] = lds[i].compute(ldsResults[i]);
        }
        sc.gradientCheck(ldsResults[lds.length], labels);
        CostResult cr = sc.stackedCost(ldsResults[lds.length], labels);
        DoubleMatrix delta = cr.delta;
        for(int k = lds.length-1; k >= 0; k--) {
            cr = lds[k].cost(ldsResults[k], ldsResults[k+1], delta);
            delta = cr.delta;
            lds[k].gradientCheck(input, labels, cr, this);
        }
        DoubleMatrix[][] delt = expand(delta, convResults[cls.length][0].length, convResults[cls.length][0][0].rows, convResults[cls.length][0][0].columns);
        for(int k = cls.length-1; k >= 0; k--) {
            delt = cls[k].gradientCheck(input, labels, convResults[k], convResults[k+1], delt, this);
        }
    }



    private DoubleMatrix[][] expand(DoubleMatrix in, int channels, int rows, int cols) {
        DoubleMatrix[][] result = new DoubleMatrix[in.rows][channels];
        for(int i = 0; i < in.rows; i++) {
            for(int j = 0; j < channels; j++) {
                result[i][j] = new DoubleMatrix(rows, cols);
                for(int k = 0; k < rows; k++) {
                    for(int l = 0; l < cols; l++) {
                        result[i][j].put(k, l, in.get(i, j * rows * cols + k * cols + l));
                    }
                }
            }
        }
        return result;
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

    public DoubleMatrix compute(DoubleMatrix[][] input) {
        for(int i = 0; i < cls.length; i++) {
            input = cls[i].compute(input);
        }
        DoubleMatrix in = Utils.flatten(input);
        for(int i = 0; i < lds.length; i++) {
            in = lds[i].compute(in);
        }
        in = sc.compute(in);
        return in;
    }
}
