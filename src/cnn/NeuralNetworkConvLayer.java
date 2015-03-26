package cnn;

import org.jblas.DoubleMatrix;

import java.io.IOException;

/**
 * Created by jassmanntj on 3/25/2015.
 */
public abstract class NeuralNetworkConvLayer {
    public abstract DoubleMatrix[][] compute(DoubleMatrix input[][]);
    public abstract DoubleMatrix[][] train(DoubleMatrix input[][], int iterations) throws IOException;
    public abstract DoubleMatrix[][] backPropagation(DoubleMatrix[][] results, DoubleMatrix[][] delta, double momentum, double alpha);
    public abstract void writeLayer(String filename);
    public abstract void loadLayer(String filename);
}
