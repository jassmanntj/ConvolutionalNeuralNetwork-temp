package cnn.device;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

public class DeviceCNN {
	DeviceNeuralNetworkLayer[] layers;

	public DeviceCNN(DeviceNeuralNetworkLayer[] layers) {
		this.layers = layers;
	}
	
	public int[] compute(DoubleMatrix2D input) {
		for(int i = 0; i < layers.length; i++) {
			input = layers[i].compute(input);
		}
		return DeviceUtils.computeResults(input);
	}
}
