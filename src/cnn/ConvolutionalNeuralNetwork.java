package cnn;

import java.io.File;
import java.io.IOException;

import org.jblas.DoubleMatrix;

public class ConvolutionalNeuralNetwork {
	private NeuralNetworkLayer[] layers;
	private int startLayer;
	private String name;
	
	public ConvolutionalNeuralNetwork(NeuralNetworkLayer[] layers, String name) {
		this.layers = layers;
		this.name = name;
		startLayer = 0;
		for(int i = 0; i < layers.length; i++) {
			if(layers[i] instanceof ConvolutionLayer) startLayer = i;
		}
	}
	
	public DoubleMatrix train(DoubleMatrix input, DoubleMatrix labels, int iterations) throws IOException {
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
