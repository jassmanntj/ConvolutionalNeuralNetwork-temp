package cnn;

import org.jblas.DoubleMatrix;

/**
 * Created by Tim on 4/1/2015.
 */
public class PoolingLayer extends ConvPoolLayer {
    private int poolDim;
    public PoolingLayer(int poolDim) {
        this.poolDim = poolDim;
    }

    public DoubleMatrix[][] compute(DoubleMatrix[][] in) {
        DoubleMatrix[][] result = new DoubleMatrix[in.length][in[0].length];
        for(int i = 0; i < in.length; i++) {
            for(int j = 0; j < in[i].length; j++) {
                result[i][j] = pool(in[i][j]);
            }
        }
        return result;
    }

    public DoubleMatrix[][] backPropagation(DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix delta[][], double momentum, double alpha) {
        DoubleMatrix[][] result = new DoubleMatrix[delta.length][delta[0].length];
        for(int i = 0; i < delta.length; i++) {
            for(int j = 0; j < delta[i].length; j++) {
                result[i][j] = expand(delta[i][j]);
            }
        }
        return result;
    }

    public DoubleMatrix[][] gradientCheck(DoubleMatrix[][] in, DoubleMatrix labels, DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix[][] delta, NeuralNetwork cnn) {
        DoubleMatrix[][] result = new DoubleMatrix[delta.length][delta[0].length];
        for(int i = 0; i < delta.length; i++) {
            for(int j = 0; j < delta[i].length; j++) {
                result[i][j] = expand(delta[i][j]);
            }
        }
        return result;
    }

        private DoubleMatrix pool(DoubleMatrix convolvedFeature) {
        int resultRows = convolvedFeature.rows/poolDim;
        int resultCols = convolvedFeature.columns/poolDim;
        DoubleMatrix result = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDim, poolRow*poolDim+poolDim, poolCol*poolDim, poolCol*poolDim+poolDim);
                result.put(poolRow, poolCol, patch.mean());
            }
        }
        return result;
    }

    private DoubleMatrix expand(DoubleMatrix in) {
        if(poolDim > 1) {
            DoubleMatrix expandedMatrix = new DoubleMatrix(in.rows * poolDim, in.columns * poolDim);
            double scale = (poolDim * poolDim);
            for (int i = 0; i < in.rows; i++) {
                for (int j = 0; j < in.columns; j++) {
                    double value = in.get(i, j) / scale;
                    for (int k = 0; k < poolDim; k++) {
                        for (int l = 0; l < poolDim; l++) {
                            expandedMatrix.put(i * poolDim + k, j * poolDim + l, value);
                        }
                    }
                }
            }
            return expandedMatrix;
        }
        else return in;
    }
}
