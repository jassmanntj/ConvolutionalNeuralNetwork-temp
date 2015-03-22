package cnn;

import java.io.IOException;

import numerical.LBFGS.ExceptionWithIflag;

import org.jblas.DoubleMatrix;

public class NeuralNetwork {
	private SparseAutoencoder sae;
	private SoftmaxClassifier sc;
	
	public NeuralNetwork(SparseAutoencoder sae, SoftmaxClassifier sc) {
		this.sae = sae;
		this.sc = sc;
	}
	
	public void train(DoubleMatrix input, DoubleMatrix labels, DoubleMatrix unlabeledData, int iterations) throws ExceptionWithIflag, IOException {
		sae.train(unlabeledData, unlabeledData, iterations);
		DoubleMatrix activations = sae.compute(input);
		sc.train(activations, labels, iterations);
	}
	
	public int[] compute(DoubleMatrix input) {
		DoubleMatrix features = sae.compute(input);
		return sc.computeResults(features);
	}
}
