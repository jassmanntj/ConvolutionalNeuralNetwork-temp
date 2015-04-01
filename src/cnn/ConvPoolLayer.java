package cnn;

import org.jblas.DoubleMatrix;

/**
 * Created by Tim on 4/1/2015.
 */
public abstract class ConvPoolLayer {
    public abstract DoubleMatrix[][] compute(DoubleMatrix[][] in);

    public abstract DoubleMatrix[][] backPropagation(DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix delta[][], double momentum, double alpha);

    public abstract DoubleMatrix[][] gradientCheck(DoubleMatrix[][] in, DoubleMatrix labels, DoubleMatrix[][] input, DoubleMatrix[][] output, DoubleMatrix[][] delta, NeuralNetwork cnn);
}