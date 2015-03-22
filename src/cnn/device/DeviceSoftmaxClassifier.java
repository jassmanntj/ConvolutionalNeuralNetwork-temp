package cnn.device;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import cern.colt.matrix.tdouble.DoubleMatrix2D;
import cern.colt.matrix.tdouble.impl.DenseDoubleMatrix2D;

public class DeviceSoftmaxClassifier extends DeviceNeuralNetworkLayer {
	private DenseDoubleMatrix2D theta;
	
	public DeviceSoftmaxClassifier(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DenseDoubleMatrix2D(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			for(int i = 0; i < theta.rows(); i++) {
				for(int j = 0; j < theta.columns(); j++) {
					theta.set(i, j, Double.parseDouble(data[i*theta.columns()+j]));

				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public DenseDoubleMatrix2D compute(DoubleMatrix2D input) {
		DoubleMatrix2D result = new DenseDoubleMatrix2D(input.rows(), theta.columns());
		return (DenseDoubleMatrix2D) input.zMult(theta, result);
	}

	@Override
	public DoubleMatrix2D getTheta() {
		return theta;
	}

	@Override
	public DoubleMatrix2D getBias() {
		return null;
	}

	@Override
	public void loadTheta(String filename) {
		try {
			FileReader fr = new FileReader(filename);
			@SuppressWarnings("resource")
			BufferedReader reader = new BufferedReader(fr);
			String[] thetaSize = reader.readLine().split(",");
			theta = new DenseDoubleMatrix2D(Integer.parseInt(thetaSize[0]),Integer.parseInt(thetaSize[1]));
			String[] data = reader.readLine().split(",");
			for(int i = 0; i < theta.columns(); i++) {
				for(int j = 0; j < theta.rows(); j++) {
					theta.set(j, i, Double.parseDouble(data[i*theta.rows()+j]));

				}
			}
		}
		catch(IOException e) {
			e.printStackTrace();
		}
		
	}

}
