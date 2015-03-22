package cnn.device;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class DeviceLinearDecoder extends DeviceNeuralNetworkLayer{
	private DenseDoubleMatrix2D theta1;
	private DenseDoubleMatrix2D bias1;
	
	public DeviceLinearDecoder(int inputSize, int hiddenSize, String filename) {
		theta1 = new DenseDoubleMatrix2D(inputSize, hiddenSize);
		bias1 = new DenseDoubleMatrix2D(1, hiddenSize);
		loadTheta(filename);
	}
	
	public DeviceLinearDecoder(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] data = reader.readLine().split(",");
			theta1 = new DenseDoubleMatrix2D(Integer.parseInt(data[0]), Integer.parseInt(data[1]));
			data = reader.readLine().split(",");
			for(int i = 0; i < theta1.rows(); i++) {
				for(int j = 0; j < theta1.columns(); j++) {
					theta1.set(i, j, Double.parseDouble(data[i*theta1.columns()+j]));
				}
			}
			data = reader.readLine().split(",");
			bias1 = new DenseDoubleMatrix2D(Integer.parseInt(data[0]), Integer.parseInt(data[1]));
			for(int i = 0; i < bias1.rows(); i++) {
				for(int j = 0; j < bias1.columns(); j++) {
					bias1.set(i, j, Double.parseDouble(data[i*theta1.columns()+j]));
				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public DoubleMatrix2D compute(DoubleMatrix2D input) {
		return null;
	}

	@Override
	public DoubleMatrix2D getTheta() {
		return theta1;
	}

	@Override
	public DoubleMatrix2D getBias() {
		return bias1;
	}

	@Override
	public void loadTheta(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] data = reader.readLine().split(",");
			for(int i = 0; i < theta1.columns(); i++) {
				for(int j = 0; j < theta1.rows(); j++) {
					theta1.set(j, i, Double.parseDouble(data[i*theta1.rows()+j]));

				}
			}
			data = reader.readLine().split(",");
			for(int i = 0; i < bias1.columns(); i++) {
				for(int j = 0; j < bias1.rows(); j++) {
					bias1.set(j, i, Double.parseDouble(data[i*bias1.rows()+j]));

				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
}
