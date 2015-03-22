package cnn.device;

import cern.colt.matrix.tdouble.DoubleMatrix2D;

public abstract class DeviceNeuralNetworkLayer {
	public abstract DoubleMatrix2D compute(DoubleMatrix2D input);
	public abstract DoubleMatrix2D getTheta();
	public abstract DoubleMatrix2D getBias();
	public abstract void loadTheta(String filename);
}
