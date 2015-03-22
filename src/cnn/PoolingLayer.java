package cnn;

import org.jblas.DoubleMatrix;

import java.io.IOException;

/**
 * Created by jassmanntj on 3/22/2015.
 */
public class PoolingLayer {
    /*private int poolDimX;
    private int poolDimY;
    private int resultRows;
    private int resultCols;
    private int channels;

    public DoubleMatrix getA() {
        return null;
    }


    public PoolingLayer(int poolDim, int features) {
        this.poolDimX = poolDim;
        this.poolDimY = poolDim;
    }

    public PoolingLayer(int poolDimX, int poolDimY, int features) {
        this.poolDimX = poolDimX;
        this.poolDimY = poolDimY;
    }

    public DataContainer feedForward(DataContainer input) {
        return compute(input);
    }

    @Override
    public DoubleMatrix backPropagation(DataContainer[] results, int layer, DoubleMatrix y, double momentum, double alpha) {
        DoubleMatrix delta = expand(y);
        return delta;
    }

    public DataContainer compute(DataContainer in) {
        DoubleMatrix[][] input = in.getDataArray();
        this.resultRows = input[0][0].rows/poolDimY;
        this.resultCols = input[0][0].columns/poolDimX;
        this.channels = input[0].length;
        DoubleMatrix pooledFeatures[][] = new DoubleMatrix[input.length][input[0].length];

        for(int imageNum = 0; imageNum < input.length; imageNum++) {
            for(int featureNum = 0; featureNum < input[imageNum].length; featureNum++) {
                pooledFeatures[imageNum][featureNum] = pool(input[imageNum][featureNum]);
            }
        }
        return new DataContainer(pooledFeatures);
    }

    public DoubleMatrix pool(DoubleMatrix convolvedFeature) {
        DoubleMatrix pooledFeature = new DoubleMatrix(resultRows, resultCols);
        for(int poolRow = 0; poolRow < resultRows; poolRow++) {
            for(int poolCol = 0; poolCol < resultCols; poolCol++) {
                DoubleMatrix patch = convolvedFeature.getRange(poolRow*poolDimX, poolRow*poolDimX+poolDimX, poolCol*poolDimY, poolCol*poolDimY+poolDimY);
                pooledFeature.put(poolRow, poolCol, patch.mean());
            }
        }
        return pooledFeature;
    }

    private DoubleMatrix expand(DoubleMatrix in) {
        System.out.println(in.rows+"::"+in.columns);
        DoubleMatrix expandedMatrix = new DoubleMatrix(in.rows*poolDimX, in.columns*poolDimY);
        int srcRows = resultRows * poolDimY;
        int srcCols = resultCols * poolDimX;
        double scale = (poolDimX * poolDimY);
        for(int i = 0; i < in.rows; i++) {
            for(int j = 0; j < this.channels; j++) {
                for(int k = 0; k < resultRows; k++) {
                    for(int l = 0; l < resultCols; l++) {
                        double value = in.get(k, j*resultRows*resultCols+k*resultCols+j)/scale;
                        for(int m = 0; m < poolDimY; m++) {
                            for(int n = 0; n < poolDimX; n++) {
                                expandedMatrix.put(i, j*srcRows*srcCols+(k+m)*srcCols+l+n, value);
                            }
                        }
                    }
                }
            }
        }
        return expandedMatrix;
    }

    public DoubleMatrix getTheta() {
        return null;
    }
    public DoubleMatrix getBias() {
        return null;
    }
    public void writeTheta(String filename) {
    }
    public DataContainer train(DataContainer input, DoubleMatrix output, int iterations) throws IOException {
        return compute(input);
    }
    public void writeLayer(String filename) {
    }
    public void loadLayer(String filename) {
    }*/
}
