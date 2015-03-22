package cnn;

import java.io.IOException;

import org.jblas.DoubleMatrix;

public abstract class NeuralNetworkLayer {
	public abstract DoubleMatrix compute(DoubleMatrix input);
	public abstract DoubleMatrix getTheta();
	public abstract DoubleMatrix getBias();
	public abstract DoubleMatrix train(DoubleMatrix input, DoubleMatrix output, int iterations) throws IOException;
	public abstract void writeLayer(String filename);
	public abstract void loadLayer(String filename);

    public abstract DoubleMatrix feedForward(DoubleMatrix input);
}
